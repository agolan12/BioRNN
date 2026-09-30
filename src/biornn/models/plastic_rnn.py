"""
A plastic RNN: static (memoryless) nodes, dynamic weights. The recurrent
connectivity is a fixed, gradient-trained backbone W_hh plus a fast plastic
component A that evolves *within each sequence* under the associative
(Hebbian) rule of Wakhloo, Clark & Abbott (2025), "Associative synaptic
plasticity creates dynamic persistent activity" -- see
papers/wakhloo2025/persistent_oscillations.py for the continuous-time model:

    dx_i/dt    = -x_i + sum_j (J_ij + A_ij) phi(x_j) + h_i(t)          (Eq. 1)
    p dA_ij/dt = -A_ij + (k/N) phi(x_i) phi(x_j)                      (Eq. 3)

Euler-discretizing Eq. 1 with dt equal to the neuron time constant (dt = tau)
removes the leak entirely and gives exactly the vanilla Elman update in
`rnn.py` -- that is what "static nodes" means here: a unit's state depends on
the past only through its recurrent input, with no self-memory. All the slow
dynamics live in the synapses. Discretizing Eq. 3 with the same step gives

    h_t = tanh(W_xh x_t + b + (W_hh + A_{t-1}) h_{t-1})
    A_t = (1 - 1/p) A_{t-1} + (k / (p N)) h_t h_t^T,       A_0 = 0

with W_hh playing the role of the paper's random backbone J, h = phi(x) the
firing rates, p the synaptic timescale in timesteps (p >= 1 keeps the Euler
decay factor in [0, 1)), and k the plasticity strength (k > 0 Hebbian).

A is a per-sequence *state*, not a parameter: it starts at zero for every
sequence (as in the paper) and is never touched by the optimizer. Gradient
descent trains W_xh, b, W_hh and the readout; backprop flows through A's
trajectory, so the backbone learns to use plasticity. k = 0 recovers the
vanilla RNN exactly. The one-step Jacobian gains a slow path through A:

    dA_t/dA_{t-1} = (1 - 1/p) I      (plus the Hebbian term via h_t),

so information written into A decays like (1 - 1/p)^k rather than through
repeated multiplication by the recurrent Jacobian.
"""

import torch
import torch.nn as nn

from .rnn import VanillaRNNCell, AddingRNN


class PlasticRNNCell(VanillaRNNCell):
    """
    One plastic-RNN update, (x_t, h_{t-1}, A_{t-1}) -> (h_t, A_t).

    Subclasses VanillaRNNCell so `set_spectral_radius` still applies to the
    backbone W_hh. `learn_k=True` makes the plasticity strength k a trainable
    scalar (initialized at `k`); p is always fixed.
    """

    def __init__(self, input_size, hidden_size, k=2.8, p=25.0, learn_k=False):
        super().__init__(input_size, hidden_size)
        assert p >= 1.0, "p < 1 makes the Euler decay factor (1 - 1/p) negative"
        self.p = p
        if learn_k:
            self.k = nn.Parameter(torch.tensor(float(k)))
        else:
            self.register_buffer("k", torch.tensor(float(k)))

    def init_plastic(self, batch, device=None, dtype=None):
        """A_0 = 0: no plastic component at the start of a sequence."""
        return torch.zeros(batch, self.hidden_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x_t, h_prev, A_prev):
        # A_ij is post i <- pre j, so the plastic recurrent input is A @ h.
        plastic_input = torch.einsum("bij,bj->bi", A_prev, h_prev)
        h = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev) + plastic_input)
        hebb = torch.einsum("bi,bj->bij", h, h)  # phi(x_i) phi(x_j), post x pre
        A = (1 - 1 / self.p) * A_prev + (self.k / (self.p * self.hidden_size)) * hebb
        return h, A


class PlasticAddingRNN(AddingRNN):
    """
    Plastic RNN for the (frequency-)Adding Problem: consumes a
    (batch, seq_len, 2) sequence and regresses a scalar from the final
    hidden state, like AddingRNN.

    `return_plastic=True` additionally returns the list of A_t, each
    (batch, hidden, hidden); keep the batch small when inspecting it.
    """

    def __init__(self, hidden_size, input_size=2, k=2.8, p=25.0, learn_k=False):
        # Build modules in the same order as AddingRNN (skipping its own cell)
        # so that a given torch seed yields identical W_xh, W_hh and readout,
        # making k=0 reproduce the vanilla model exactly.
        nn.Module.__init__(self)
        self.cell = PlasticRNNCell(input_size, hidden_size, k=k, p=p, learn_k=learn_k)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x, return_hidden_states=False, return_plastic=False):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        A = self.cell.init_plastic(batch, device=x.device, dtype=x.dtype)
        hidden_states, plastic_states = [], []
        for t in range(seq_len):
            h, A = self.cell(x[:, t, :], h, A)
            if return_hidden_states:
                if h.requires_grad:  # not under torch.no_grad()
                    h.retain_grad()
                hidden_states.append(h)
            if return_plastic:
                plastic_states.append(A)
        out = self.readout(h).squeeze(-1)
        extras = []
        if return_hidden_states:
            extras.append(hidden_states)
        if return_plastic:
            extras.append(plastic_states)
        return (out, *extras) if extras else out
