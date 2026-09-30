"""
Plotting utilities for the vanishing/exploding gradient demo:

  - plot_rnn_diagram   : draws the network itself (folded cell + unrolled
                          computation graph), so you can see exactly which
                          weight matrix is reused at every timestep.
  - plot_gradient_norms: log-scale norm of the backpropagated gradient as a
                          function of time-lag -- the classic exponential
                          vanishing/exploding curve.
  - plot_training_curve: plain loss-vs-epoch curves for the benchmark runs.
"""

import numpy as np
import matplotlib.pyplot as plt

_H_COLOR = "#8ecae6"
_X_COLOR = "#ffb703"
_Y_COLOR = "#fb8500"
_REC_COLOR = "#c1121f"


def _draw_folded(ax):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.9, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Folded RNN cell")

    r = 0.4
    ax.add_patch(plt.Circle((0, 0), r, fc=_H_COLOR, ec="black", zorder=3))
    ax.text(0, 0, "$h_t$", ha="center", va="center", fontsize=13, zorder=4)

    ax.add_patch(plt.Circle((0, -1.3), 0.32, fc=_X_COLOR, ec="black", zorder=3))
    ax.text(0, -1.3, "$x_t$", ha="center", va="center", fontsize=12, zorder=4)
    ax.annotate("", xy=(0, -r), xytext=(0, -1.3 + 0.32),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.35, -0.75, "$W_{xh}$", fontsize=11)

    ax.add_patch(plt.Circle((0, 1.3), 0.32, fc=_Y_COLOR, ec="black", zorder=3))
    ax.text(0, 1.3, "$y_t$", ha="center", va="center", fontsize=12, zorder=4)
    ax.annotate("", xy=(0, 1.3 - 0.32), xytext=(0, r),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.35, 0.75, "$W_{hy}$", fontsize=11)

    # self-recurrent loop, bulging to the left of h_t
    ax.annotate(
        "", xy=(-r * 0.75, r * 0.65), xytext=(-r * 0.75, -r * 0.65),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=_REC_COLOR,
                         connectionstyle="arc3,rad=1.5"),
    )
    ax.text(-1.35, 0, "$W_{hh}$", fontsize=11, color=_REC_COLOR)


def _draw_unrolled(ax, steps):
    r = 0.26
    dx = 1.5
    ax.set_xlim(-1, (steps - 1) * dx + 1)
    ax.set_ylim(-1.9, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Unrolled through time ({steps} steps)")

    for t in range(steps):
        cx = t * dx

        ax.add_patch(plt.Circle((cx, -1.2), r, fc=_X_COLOR, ec="black", zorder=3))
        ax.text(cx, -1.2, f"$x_{{{t+1}}}$", ha="center", va="center", fontsize=9, zorder=4)

        ax.add_patch(plt.Circle((cx, 0), r, fc=_H_COLOR, ec="black", zorder=3))
        ax.text(cx, 0, f"$h_{{{t+1}}}$", ha="center", va="center", fontsize=9, zorder=4)

        ax.add_patch(plt.Circle((cx, 1.2), r, fc=_Y_COLOR, ec="black", zorder=3))
        ax.text(cx, 1.2, f"$y_{{{t+1}}}$", ha="center", va="center", fontsize=9, zorder=4)

        ax.annotate("", xy=(cx, -r), xytext=(cx, -1.2 + r),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(cx, 1.2 - r), xytext=(cx, r),
                    arrowprops=dict(arrowstyle="->"))

        if t > 0:
            ax.annotate("", xy=(cx - r, 0), xytext=(cx - dx + r, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=_REC_COLOR))

    ax.text((steps - 1) * dx / 2, -1.75,
             r"same $W_{xh}$, $W_{hh}$, $W_{hy}$ reused at every timestep",
             ha="center", fontsize=9, style="italic")


def plot_rnn_diagram(unroll_steps=5, figsize=(13, 5)):
    """Draw the folded RNN cell next to its unrolled-through-time graph."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _draw_folded(ax1)
    _draw_unrolled(ax2, unroll_steps)
    fig.suptitle("Vanilla (Elman) RNN architecture", fontsize=14)
    fig.tight_layout()
    return fig


def plot_gradient_norms(curves, title="Gradient norm vs. time-lag",
                         ylabel=r"$\|\partial h_T / \partial h_t\|$",
                         xlabel="time lag  (T - t)"):
    """
    curves: dict {label -> 1D array of gradient norms indexed by lag 0..T-1}
    Plotted on a log y-axis; a horizontal reference line at 1.0 separates the
    vanishing regime (below) from the exploding regime (above).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, norms in curves.items():
        norms = np.asarray(norms)
        ax.plot(np.arange(len(norms)), norms, marker="o", markersize=3, label=label)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_curve(curves, title="Training loss", ylabel="loss", logy=True):
    """curves: dict {label -> list/array of per-epoch values}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in curves.items():
        ax.plot(values, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison_bars(categories, baseline, fixed, ylabel="final loss",
                          title="Baseline vs. fixed", baseline_label="baseline (no fixes)",
                          fixed_label="clipping + norm-preserving regularizer",
                          reference=None, reference_label=None):
    """
    Grouped bar chart comparing a `baseline` value against a `fixed` value
    for each entry in `categories` (e.g. one bar pair per sequence length or
    delay). `reference` optionally draws a horizontal dashed line (e.g. the
    unconditional-mean baseline for a regression task, or chance level for a
    classification task).
    """
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, baseline, width, label=baseline_label, color="#c1121f")
    ax.bar(x + width / 2, fixed, width, label=fixed_label, color="#219ebc")
    if reference is not None:
        ax.axhline(reference, color="gray", linestyle="--", linewidth=1,
                    label=reference_label or "reference")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
