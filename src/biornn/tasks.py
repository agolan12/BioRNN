"""
Synthetic long-range-dependency benchmarks classically used to expose the
vanishing/exploding gradient problem in vanilla RNNs:

  - Adding Problem      (Hochreiter & Schmidhuber, 1997; Pascanu et al., 2013)
  - Frequency Adding Problem (variant of the Adding Problem, see below)
  - Copy Memory Task    (Hochreiter & Schmidhuber, 1997; Arjovsky et al., 2016)

Every task generates its data with NumPy on the CPU (so a seeded `rng` gives
the same examples on any device) and returns tensors moved to `device`
(default: CPU).

All tasks require the network to carry information across a gap of `seq_len`
(or `delay`) timesteps, which is exactly the regime where BPTT gradients
through a vanilla RNN vanish (or explode) exponentially in the gap length.

`frequency_adding_problem` is a variant of the classic Adding Problem in which
the two random numbers to be added are not presented directly as scalars but
*encoded in the frequency of a windowed cosine oscillation*, in the spirit of
the oscillatory input h_i(t) = I * cos(2*pi*f*t + theta_i) used in Wakhloo,
Clark & Abbott (2025) ("Associative synaptic plasticity creates dynamic
persistent activity"), see papers/wakhloo2025/persistent_oscillations.py,
Eq. 2. Here the network itself must decode a frequency from a short
oscillatory snippet before it can even attempt the addition -- a strictly
harder perceptual front-end bolted onto the same long-range
credit-assignment problem.
"""

import numpy as np
import torch


def adding_problem(batch_size, seq_len, rng=None, device=None):
    """
    Each example is a length-`seq_len` sequence of (value, marker) pairs.
    `value` is drawn uniformly from [0, 1]; exactly two timesteps are
    "marked" (marker = 1), one in the first half and one in the second half
    of the sequence. The target is the sum of the two marked values.

    A network that cannot propagate gradient/information across the gap
    between the two markers cannot do better than always predicting the
    unconditional mean of the target (~1.0).

    Returns:
        x: FloatTensor (batch_size, seq_len, 2)
        y: FloatTensor (batch_size,)
    """
    rng = rng or np.random.default_rng()
    values = rng.uniform(0, 1, size=(batch_size, seq_len)).astype(np.float32)
    markers = np.zeros((batch_size, seq_len), dtype=np.float32)

    half = seq_len // 2
    first_idx = rng.integers(0, half, size=batch_size)
    second_idx = rng.integers(half, seq_len, size=batch_size)
    rows = np.arange(batch_size)
    markers[rows, first_idx] = 1.0
    markers[rows, second_idx] = 1.0

    x = np.stack([values, markers], axis=-1)
    y = (values * markers).sum(axis=1)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def frequency_adding_problem(batch_size, seq_len, window_len=5, rng=None, device=None):
    """
    Each of the two random values f1, f2 ~ Uniform(0, 1) is encoded as the
    frequency of a windowed cosine snippet:

        signal[t] = cos(pi * f * (t - t0))   for t in [t0, t0 + window_len)

    using angular argument `pi * f` (not the more familiar `2*pi*f`) so that
    f=1 lands exactly at the discrete-time Nyquist frequency (alternating
    +1/-1 samples) rather than aliasing back down to f=0 -- with a plain
    `2*pi*f` argument, integer-sample cosines of frequency f and frequency
    f+1 (or 1-f) would be indistinguishable. This keeps every f in [0, 1]
    uniquely decodable from evenly-spaced integer samples.

    One window is placed uniformly at random in the first half of the
    sequence and the other in the second half (mirroring the marker
    placement of the original Adding Problem), so the network must still
    carry information across the gap between them. A second input channel
    marks which timesteps belong to a window, since the cosine signal itself
    legitimately passes through zero and can't be distinguished from
    "no signal" on its own.

    Returns:
        x: FloatTensor (batch_size, seq_len, 2)  -- (signal, marker) channels
        y: FloatTensor (batch_size,)             -- target f1 + f2
    """
    rng = rng or np.random.default_rng()
    half = seq_len // 2
    assert half > window_len, "seq_len must be more than 2*window_len"

    freqs = rng.uniform(0, 1, size=(batch_size, 2)).astype(np.float32)
    start1 = rng.integers(0, half - window_len, size=batch_size)
    start2 = rng.integers(half, seq_len - window_len, size=batch_size)

    signal = np.zeros((batch_size, seq_len), dtype=np.float32)
    marker = np.zeros((batch_size, seq_len), dtype=np.float32)
    taps = np.arange(window_len, dtype=np.float32)

    for i in range(batch_size):
        s1, s2 = start1[i], start2[i]
        signal[i, s1:s1 + window_len] = np.cos(np.pi * freqs[i, 0] * taps)
        signal[i, s2:s2 + window_len] = np.cos(np.pi * freqs[i, 1] * taps)
        marker[i, s1:s1 + window_len] = 1.0
        marker[i, s2:s2 + window_len] = 1.0

    x = np.stack([signal, marker], axis=-1)
    y = freqs.sum(axis=1)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def copy_task(batch_size, pattern_len, delay, n_categories=8, rng=None, device=None):
    """
    A length-`pattern_len` random pattern (symbols 1..n_categories) appears
    at the start of the sequence, followed by `delay` blank steps (symbol 0),
    a single delimiter step (symbol n_categories + 1), and finally
    `pattern_len` more blank steps during which the network must reproduce
    the original pattern in order.

    Total sequence length = pattern_len + delay + 1 + pattern_len.

    Returns:
        x: LongTensor (batch_size, total_len)   -- input symbol indices
        y: LongTensor (batch_size, total_len)   -- target symbol indices
           (target is 0/blank everywhere except the final pattern_len steps)
    """
    rng = rng or np.random.default_rng()
    delimiter = n_categories + 1
    pattern = rng.integers(1, n_categories + 1, size=(batch_size, pattern_len))

    total_len = pattern_len + delay + 1 + pattern_len
    x = np.zeros((batch_size, total_len), dtype=np.int64)
    y = np.zeros((batch_size, total_len), dtype=np.int64)

    x[:, :pattern_len] = pattern
    x[:, pattern_len + delay] = delimiter
    y[:, -pattern_len:] = pattern

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
