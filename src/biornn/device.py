"""
Pick the torch device to train on.

Order of preference: an explicit `name` argument, then the BIORNN_DEVICE
environment variable, then the best available accelerator (CUDA, then Apple
Metal / MPS), then CPU. The environment variable lets you switch every
notebook at once without editing it, e.g. `BIORNN_DEVICE=cpu jupyter lab`.

A GPU is not automatically faster here. The models loop over time in Python
and launch a few small kernels per step, so at hidden sizes <= 64 the
per-launch overhead dominates: on an M3, vanilla/leaky training was ~2x
slower on MPS than on CPU, while the plastic RNN (whose per-step
(batch, N, N) Hebbian update is real work) was ~1.7x faster.
"""

import os

import torch


def get_device(name=None):
    name = name or os.environ.get("BIORNN_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
