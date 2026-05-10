"""
Weight-level network bending utilities for PLAUD.

All operations mutate the model **in place** using `param.data.copy_()`,
which is the correct approach for TorchScript modules where direct assignment
(`param = new_tensor`) raises an error.

Always operate on a freshly-loaded copy:
    m = torch.jit.load(path, map_location='cpu').eval()
so the original checkpoint on disk stays clean.
"""

from __future__ import annotations
import torch


def _find_param(model, name: str) -> torch.nn.Parameter:
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise KeyError(f"parameter '{name}' not found in model")


def scale(model, param_name: str, factor: float) -> None:
    """Multiply every weight value by `factor`."""
    p = _find_param(model, param_name)
    p.data.copy_(p.data * factor)


def shift(model, param_name: str, offset: float) -> None:
    """Add `offset` to every weight value."""
    p = _find_param(model, param_name)
    p.data.copy_(p.data + offset)


def add_noise(model, param_name: str, std: float, seed: int | None = None) -> None:
    """Add zero-mean Gaussian noise to a weight tensor."""
    p = _find_param(model, param_name)
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(p.data) * std
    p.data.copy_(p.data + noise)


def zero_out(model, param_name: str) -> None:
    """Zero all values in a weight tensor (drastic — ablation / kill-switch)."""
    p = _find_param(model, param_name)
    p.data.zero_()


def roll(model, param_name: str, shifts: int, dim: int = 0) -> None:
    """Cyclically shift weight values along `dim` by `shifts` positions.

    Bending-log effect: displaces which neuron connects to which, causing
    timbral smearing, filtering-like spectral changes, and transient effects.
    """
    p = _find_param(model, param_name)
    p.data.copy_(torch.roll(p.data, shifts, dims=dim))


def flip(model, param_name: str, dim: int = 0) -> None:
    """Reverse the order of weight values along `dim`.

    Bending-log effect on input_bottleneck.4.weight: adds a lot of highs.
    """
    p = _find_param(model, param_name)
    p.data.copy_(torch.flip(p.data, [dim]))


def squeeze(model, param_name: str, factor: float = 0.5) -> None:
    """Compress weight values toward zero (equivalent to scale with factor < 1).

    Bending-log effect: sharper, more compressed sound.
    """
    scale(model, param_name, factor)
