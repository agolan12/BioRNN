"""
Sweep vanilla vs. plastic RNNs on the frequency-encoded (oscillating-input)
Adding Problem over hidden size x sequence length, then compare them.

Training matches `train_adding` / `test_mse` in
notebooks/02_frequency_adding/04_plastic_rnn.ipynb (Adam, fresh batch of 128
per epoch, held-out MSE on 1024 sequences from seed 12345), except for the
epoch count and the grid.

Subcommands:

    count    print the number of runs in the grid (used to size the SLURM array)
    run      train one grid point; the index comes from --task-id or
             $SLURM_ARRAY_TASK_ID. Skips runs whose result file already exists,
             so a preempted/requeued array can simply be resubmitted.
    compare  collect every result in --out-dir into results.csv + figures

Each run writes <out-dir>/runs/<name>.json (metrics, subsampled loss curve)
and <name>.pt (trained weights).

Vanilla uses AddingRNN, not PlasticAddingRNN(k=0): same weights for a given
seed, but no (batch, N, N) plastic state, which is what dominates GPU memory
at large N. If a full batch still doesn't fit, the step is split into chunks
with gradient accumulation (identical gradient, just slower).
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = ("vanilla", "plastic")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["count", "run", "compare"])
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "plastic_vs_vanilla")
    # grid
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=list(range(20, 201, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    # training (fixed across the grid)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--k", type=float, default=2.8, help="plasticity strength (Wakhloo et al. 2025)")
    parser.add_argument("--p", type=float, default=25.0, help="synaptic timescale in timesteps")
    parser.add_argument("--learn-k", action="store_true")
    return parser.parse_args()


def grid(args):
    return list(itertools.product(args.models, args.hidden_sizes, args.seq_lens, args.seeds))


def run_name(model, hidden_size, seq_len, seed):
    return f"{model}_h{hidden_size}_T{seq_len}_s{seed}"


# ---------------------------------------------------------------- training

def train_step(model, opt, x, y, chunks):
    """One optimizer step on (x, y), split into `chunks` pieces if needed."""
    import torch.nn.functional as F

    opt.zero_grad()
    total = 0.0
    for xc, yc in zip(x.chunk(chunks), y.chunk(chunks)):
        # sum / full batch size, so the accumulated gradient equals the full-batch mean
        loss = F.mse_loss(model(xc), yc, reduction="sum") / len(y)
        loss.backward()
        total += loss.item()
    opt.step()
    return total


def test_mse(model, seq_len, window_len, device, n=1024, seed=12345):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from biornn.tasks import frequency_adding_problem

    x, y = frequency_adding_problem(n, seq_len, window_len=window_len, rng=np.random.default_rng(seed), device=device)
    with torch.no_grad():
        return sum(F.mse_loss(model(xc), yc, reduction="sum").item()
                   for xc, yc in zip(x.chunk(8), y.chunk(8))) / n


def run(args):
    import numpy as np
    import torch
    from biornn.device import get_device
    from biornn.models.plastic_rnn import PlasticAddingRNN
    from biornn.models.rnn import AddingRNN
    from biornn.tasks import frequency_adding_problem

    task_id = args.task_id if args.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    model_name, hidden_size, seq_len, seed = grid(args)[task_id]
    name = run_name(model_name, hidden_size, seq_len, seed)
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    result_path = runs_dir / f"{name}.json"
    if result_path.exists():
        print(f"{name}: already done, skipping")
        return

    device = get_device()
    print(f"task {task_id}: {name} on {device}", flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if model_name == "vanilla":
        model = AddingRNN(hidden_size=hidden_size)
    else:
        model = PlasticAddingRNN(hidden_size=hidden_size, k=args.k, p=args.p, learn_k=args.learn_k)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    chunks, losses, start = 1, [], time.time()
    for epoch in range(args.epochs):
        x, y = frequency_adding_problem(args.batch_size, seq_len, window_len=args.window_len, rng=rng, device=device)
        while True:
            try:
                losses.append(train_step(model, opt, x, y, chunks))
                break
            except torch.cuda.OutOfMemoryError:
                if chunks >= args.batch_size:
                    raise
                chunks *= 2
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  OOM at epoch {epoch}: splitting each batch into {chunks} chunks", flush=True)
        if epoch % 200 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>5}  loss {losses[-1]:.4f}  ({time.time() - start:.0f}s)", flush=True)

    model.eval()
    result = {
        "model": model_name, "hidden_size": hidden_size, "seq_len": seq_len, "seed": seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "window_len": args.window_len,
        "k": args.k if model_name == "plastic" else 0.0, "p": args.p if model_name == "plastic" else None,
        "learn_k": args.learn_k and model_name == "plastic",
        "test_mse": test_mse(model, seq_len, args.window_len, device),
        "final_train_loss": float(np.mean(losses[-100:])),  # mean of last 100 epochs, less noisy than the last one
        "loss_curve_every_10": losses[::10],
        "train_seconds": time.time() - start, "chunks": chunks, "device": str(device),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    if model_name == "plastic":
        # Ablation from notebook 04 Sec. 5: same trained weights, plasticity off at test time.
        trained_k = model.cell.k.item()
        result["k_trained"] = trained_k
        with torch.no_grad():
            model.cell.k.fill_(0.0)
            result["test_mse_k0"] = test_mse(model, seq_len, args.window_len, device)
            model.cell.k.fill_(trained_k)

    torch.save(model.state_dict(), runs_dir / f"{name}.pt")
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(result_path)  # atomic: a preempted job never leaves a half-written result
    print(f"{name}: test MSE {result['test_mse']:.4f}"
          + (f", with k=0 at test {result['test_mse_k0']:.4f}" if model_name == "plastic" else ""))


# ---------------------------------------------------------------- comparison

BASELINE = 2 / 12  # Var(f1 + f2), f_i ~ U(0, 1): MSE of always predicting the mean
VANILLA_COLOR, PLASTIC_COLOR, ABLATED_COLOR = "#2a78d6", "#eb6834", "#1baf7a"


def compare(args):
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

    rows = [json.loads(p.read_text()) for p in sorted((args.out_dir / "runs").glob("*.json"))]
    if not rows:
        raise SystemExit(f"no results in {args.out_dir / 'runs'}")
    fields = ["model", "hidden_size", "seq_len", "seed", "test_mse", "test_mse_k0", "final_train_loss",
              "k", "k_trained", "p", "epochs", "lr", "batch_size", "window_len", "n_params", "train_seconds", "chunks", "device"]
    with open(args.out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["hidden_size"], r["seq_len"], r["seed"])))

    hs = sorted({r["hidden_size"] for r in rows})
    ts = sorted({r["seq_len"] for r in rows})

    def table(model, key="test_mse"):
        """(len(hs), len(ts)) array of `key`, averaged over seeds; NaN where missing."""
        out = np.full((len(hs), len(ts)), np.nan)
        for i, h in enumerate(hs):
            for j, t in enumerate(ts):
                vals = [r[key] for r in rows if r["model"] == model and r["hidden_size"] == h and r["seq_len"] == t]
                if vals:
                    out[i, j] = np.mean(vals)
        return out

    vanilla, plastic, ablated = table("vanilla"), table("plastic"), table("plastic", "test_mse_k0")
    n_missing = np.isnan(vanilla).sum() + np.isnan(plastic).sum()
    if n_missing:
        print(f"warning: {n_missing} (model, hidden, seq_len) cells have no result yet")

    # 1. heatmaps: each model's test MSE on a shared log scale, plus the log ratio
    blues = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    diverging = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f19c9b", "#b52f2f"])
    finite = np.concatenate([vanilla[np.isfinite(vanilla)], plastic[np.isfinite(plastic)]])
    mse_norm = LogNorm(vmin=max(finite.min(), 1e-5), vmax=max(finite.max(), BASELINE))
    log_ratio = np.log10(plastic / vanilla)
    lim = max(np.nanmax(np.abs(log_ratio)), 0.1) if np.isfinite(log_ratio).any() else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    extent_kw = dict(aspect="auto", origin="lower")
    for ax, data, title in [(axes[0], vanilla, "vanilla RNN"), (axes[1], plastic, f"plastic RNN (k={args.k:g}, p={args.p:g})")]:
        im = ax.imshow(data, cmap=blues, norm=mse_norm, **extent_kw)
        ax.set_title(f"{title}: test MSE")
    fig.colorbar(im, ax=axes[:2], label=f"test MSE (log scale; predict-the-mean = {BASELINE:.3f})")
    im = axes[2].imshow(log_ratio, cmap=diverging, norm=TwoSlopeNorm(0, -lim, lim), **extent_kw)
    axes[2].set_title("log10(plastic / vanilla)\nblue = plastic better, red = vanilla better")
    fig.colorbar(im, ax=axes[2], label="log10 MSE ratio")
    for ax in axes:
        ax.set_xticks(range(len(ts)), ts, rotation=90)
        ax.set_yticks(range(len(hs)), hs)
        ax.set(xlabel="sequence length", ylabel="hidden units")
    fig.savefig(args.out_dir / "heatmaps.png", dpi=150)
    plt.close(fig)

    # 2. small multiples: test MSE vs. seq_len, one panel per hidden size
    ncols = 3
    nrows = -(-len(hs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, i in zip(axes.flat, range(len(hs))):
        ax.plot(ts, vanilla[i], color=VANILLA_COLOR, lw=2, marker="o", ms=4, label="vanilla")
        ax.plot(ts, plastic[i], color=PLASTIC_COLOR, lw=2, marker="o", ms=4, label="plastic")
        ax.plot(ts, ablated[i], color=ABLATED_COLOR, lw=2, ls="--", label="plastic, k=0 at test")
        ax.axhline(BASELINE, color="gray", lw=1, ls=":", label="predict-the-mean")
        ax.set_yscale("log")
        ax.set_title(f"{hs[i]} hidden units")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(hs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("sequence length")
    for ax in axes[:, 0]:
        ax.set_ylabel("test MSE")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_dir / "mse_vs_seq_len.png", dpi=150)
    plt.close(fig)

    # 3. text summary: longest sequence each model solves (test MSE < 10% of baseline)
    solved = 0.1 * BASELINE
    print(f"\nlongest seq_len with test MSE < {solved:.4f} (10% of predict-the-mean):")
    print(f"{'hidden':>7}  {'vanilla':>8}  {'plastic':>8}")
    for i, h in enumerate(hs):
        longest = [max((t for t, m in zip(ts, tab[i]) if m < solved), default="-") for tab in (vanilla, plastic)]
        print(f"{h:>7}  {longest[0]!s:>8}  {longest[1]!s:>8}")
    print(f"\nwrote {args.out_dir / 'results.csv'}, heatmaps.png, mse_vs_seq_len.png")


def main():
    args = parse_args()
    if args.command == "count":
        print(len(grid(args)))
    elif args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
