"""
A plain vanilla (Elman) RNN, implemented with explicit weight matrices so that
every hidden state in the unrolled computation graph is directly reachable for
gradient inspection.

Intentionally minimal: no gradient clipping, no gating (LSTM/GRU), no
normalization, no skip connections, no orthogonal/identity initialization
tricks. This is the "textbook" RNN that Hochreiter (1991) and later
Bengio et al. (1994) / Pascanu et al. (2013) used to characterize the
vanishing/exploding gradient problem, kept close to that form on purpose.
"""

import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """One Elman-RNN update: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


def set_spectral_radius(cell: VanillaRNNCell, radius: float):
    """
    Rescale the recurrent weight matrix W_hh so its spectral radius
    (largest |eigenvalue|) equals `radius`.

    This is the standard knob (Pascanu et al., 2013, Sec. 2) used to move a
    vanilla RNN between the "vanishing gradient" regime (radius < 1) and the
    "exploding gradient" regime (radius > 1).
    """
    with torch.no_grad():
        w = cell.W_hh.weight.data
        # eigvals isn't implemented on MPS, so compute it on the CPU copy.
        eigs = torch.linalg.eigvals(w.cpu())
        current = eigs.abs().max().item()
        if current > 0:
            w.mul_(radius / current)


class AddingRNN(nn.Module):
    """
    Vanilla RNN for the Adding Problem: consumes a (batch, seq_len, 2)
    sequence and regresses a single scalar from the final hidden state.
    """

    def __init__(self, hidden_size, input_size=2):
        super().__init__()
        self.cell = VanillaRNNCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x, return_hidden_states=False):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        hidden_states = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            if return_hidden_states:
                h.retain_grad()
                hidden_states.append(h)
        out = self.readout(h).squeeze(-1)
        if return_hidden_states:
            return out, hidden_states
        return out


class CopyRNN(nn.Module):
    """
    Vanilla RNN for the Copy Memory Task: consumes a (batch, seq_len)
    sequence of category indices (one-hot encoded internally) and emits a
    class distribution at every timestep.
    """

    def __init__(self, hidden_size, n_categories):
        super().__init__()
        # +2 symbols: blank (0) and delimiter (n_categories + 1)
        self.n_symbols = n_categories + 2
        self.cell = VanillaRNNCell(self.n_symbols, hidden_size)
        self.readout = nn.Linear(hidden_size, self.n_symbols)
        self.hidden_size = hidden_size

    def forward(self, x, return_hidden_states=False):
        batch, seq_len = x.shape
        x_onehot = torch.nn.functional.one_hot(x, self.n_symbols).float()
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        hidden_states = []
        logits = []
        for t in range(seq_len):
            h = self.cell(x_onehot[:, t, :], h)
            if return_hidden_states:
                h.retain_grad()
                hidden_states.append(h)
            logits.append(self.readout(h))
        logits = torch.stack(logits, dim=1)  # (batch, seq_len, n_symbols)
        if return_hidden_states:
            return logits, hidden_states
        return logits


def total_grad_norm(model: nn.Module) -> float:
    """L2 norm of the gradient across all parameters (no clipping applied)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5
