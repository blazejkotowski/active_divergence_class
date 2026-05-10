"""
Activation-level network bending for PLAUD.

TorchScript modules do NOT support register_forward_hook(), so we cannot
attach hooks to submodules. Instead we implement a manual decode path that
steps through the decoder layer by layer, allowing Python code to intercept
and transform activations between layers.

The four interception points in PLAUD's decoder are:

  input_bottleneck → [*] → gru → [*] → inter_mlp → [*] → output_params → [*]

Bending at 'input_bottleneck' affects the high-level latent representation
before any recurrent processing → structural / timbral changes.
Bending at 'gru' affects the recurrent hidden state → temporal/rhythmic effects.
Bending at 'inter_mlp' shapes the pre-output feature vector → spectral effects.
Bending at 'output_params' acts right before synthesis → strong spectral reshaping.

Reference: Kotowski & Font (2026), *Network Bending as Circuit Bending
Inspired Live Neural Synthesis Hacking*, NIME.
"""

from __future__ import annotations
import math
import torch


_LN10 = math.log(10.0)

DECODE_LAYERS = ["input_bottleneck", "gru", "inter_mlp", "output_params"]


def _scaled_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Matches PLAUD's _scaled_sigmoid: 2·sigmoid(x)^ln(10) + ε."""
    return torch.sigmoid(x) ** _LN10 * 2.0 + 1e-18


def _reset_hidden(model) -> None:
    with torch.no_grad():
        model.pretrained.decoder._hidden_state.zero_()


def decode(model, z: torch.Tensor) -> torch.Tensor:
    """Synthesise audio from latent trajectory z (1, T, 4) → (1, 1, T·32).

    Equivalent to the full decoder forward pass.
    """
    _reset_hidden(model)
    with torch.no_grad():
        h0 = model.pretrained.decoder._hidden_state.clone()
        x = model.pretrained.decoder.input_bottleneck(z)
        x, _ = model.pretrained.decoder.gru.forward__0(x, h0)
        x = model.pretrained.decoder.inter_mlp(x)
        x = model.pretrained.decoder.output_params(x)
        synth_params = _scaled_sigmoid(x).permute(0, 2, 1)
        return model.pretrained._synthesize(synth_params)


def bent_decode(
    model,
    z: torch.Tensor,
    layer: str,
    transform_fn,
) -> torch.Tensor:
    """Synthesise audio with an activation transformation applied after `layer`.

    Parameters
    ----------
    model        : loaded TorchScript module
    z            : latent trajectory (1, T, 4)
    layer        : one of 'input_bottleneck', 'gru', 'inter_mlp', 'output_params'
    transform_fn : callable(activation: Tensor) → Tensor
    """
    if layer not in DECODE_LAYERS:
        raise ValueError(f"layer must be one of {DECODE_LAYERS}")

    _reset_hidden(model)
    with torch.no_grad():
        h0 = model.pretrained.decoder._hidden_state.clone()
        x = model.pretrained.decoder.input_bottleneck(z)
        if layer == "input_bottleneck":
            x = transform_fn(x)

        x, _ = model.pretrained.decoder.gru.forward__0(x, h0)
        if layer == "gru":
            x = transform_fn(x)

        x = model.pretrained.decoder.inter_mlp(x)
        if layer == "inter_mlp":
            x = transform_fn(x)

        x = model.pretrained.decoder.output_params(x)
        if layer == "output_params":
            x = transform_fn(x)

        synth_params = _scaled_sigmoid(x).permute(0, 2, 1)
        return model.pretrained._synthesize(synth_params)


# ── ready-made transform factories ──────────────────────────────────────────

def scale_fn(factor: float):
    return lambda x: x * factor

def shift_fn(offset: float):
    return lambda x: x + offset

def noise_fn(std: float, seed: int = 0):
    def fn(x):
        torch.manual_seed(seed)
        return x + torch.randn_like(x) * std
    return fn

def channel_shuffle_fn(seed: int = 0):
    def fn(x):
        torch.manual_seed(seed)
        idx = torch.randperm(x.shape[-1])
        return x[..., idx]
    return fn

def zero_fn():
    return lambda x: torch.zeros_like(x)

def roll_fn(shifts: int, dim: int = -1):
    return lambda x: torch.roll(x, shifts, dims=dim)

def flip_fn(dim: int = -1):
    return lambda x: torch.flip(x, [dim])
