"""
A standard leaky (continuous-time, Euler-discretized) RNN -- the form most
commonly used in computational neuroscience (e.g. Song, Yang & Wang, 2016;
Yang & Wang, 2020). It is derived from the rate equation

    tau * dh/dt = -h + tanh(W_xh x + W_hh h + b)

and discretized with step dt, giving

    h_t = (1 - alpha) * h_{t-1} + alpha * tanh(W_xh x_t + W_hh h_{t-1} + b),
    alpha = dt / tau.

alpha = 1 recovers the vanilla Elman RNN in `biornn.models.rnn` exactly. For alpha < 1
the one-step Jacobian is

    dh_t/dh_{t-1} = (1 - alpha) * I + alpha * diag(1 - a_t^2) W_hh,

i.e. an identity "leak" path plus a scaled-down copy of the vanilla Jacobian.
That is the whole architectural change -- no clipping, no regularizer, no
gating. `LeakyAddingRNN` / `LeakyCopyRNN` reuse the vanilla models' forward
passes unchanged and just swap in the leaky cell, so any difference in
behaviour comes from the leak alone.
"""

import torch

from .rnn import VanillaRNNCell, AddingRNN, CopyRNN


class LeakyRNNCell(VanillaRNNCell):
    """One leaky-RNN update; subclasses VanillaRNNCell so `set_spectral_radius` still applies to W_hh."""

    def __init__(self, input_size, hidden_size, alpha=0.1):
        super().__init__(input_size, hidden_size)
        self.alpha = alpha

    def forward(self, x_t, h_prev):
        return (1 - self.alpha) * h_prev + self.alpha * super().forward(x_t, h_prev)


class LeakyAddingRNN(AddingRNN):
    def __init__(self, hidden_size, input_size=2, alpha=0.1):
        super().__init__(hidden_size, input_size)
        self.cell = LeakyRNNCell(input_size, hidden_size, alpha)


class LeakyCopyRNN(CopyRNN):
    def __init__(self, hidden_size, n_categories, alpha=0.1):
        super().__init__(hidden_size, n_categories)
        self.cell = LeakyRNNCell(self.n_symbols, hidden_size, alpha)
