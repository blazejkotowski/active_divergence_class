"""
Model blending utilities for PLAUD.

Linear interpolation of weights between two PLAUD checkpoints:
    W_blend = α · W_A + (1−α) · W_B

For fully-compatible pairs (all parameter shapes match), every parameter
is blended. For partially-compatible pairs, only matching parameters are
blended; the rest fall back to model A.

TorchScript modules don't support arbitrary state-dict loading via
load_state_dict(), so blending is applied by iterating named_parameters()
and using param.data.copy_() on a freshly loaded model.
"""

from __future__ import annotations
import torch
from .compatibility import compatible_keys


def blend(
    model_a,
    model_b,
    alpha: float,
    base_path: str,
    fallback: str = "a",
) -> object:
    """Return a new model whose weights are α·A + (1−α)·B.

    Parameters
    ----------
    model_a, model_b : loaded TorchScript modules
    alpha            : blend ratio (0.0 = pure B, 1.0 = pure A)
    base_path        : path to the .ts file used as base (reload for a clean copy)
    fallback         : which model to use for non-blendable params ('a' or 'b')
    """
    m_blend = torch.jit.load(base_path, map_location='cpu').eval()

    params_a = {k: p.data.clone() for k, p in model_a.named_parameters()}
    params_b = {k: p.data.clone() for k, p in model_b.named_parameters()}
    blendable = set(compatible_keys(model_a, model_b))

    for name, p in m_blend.named_parameters():
        if name in blendable:
            blended_val = alpha * params_a[name] + (1.0 - alpha) * params_b[name]
            p.data.copy_(blended_val)
        else:
            src = params_a if fallback == "a" else params_b
            if name in src:
                p.data.copy_(src[name])

    # Copy buffers too — hidden states are excluded because decode() resets them.
    # Without this, synthesis oscillator phases (_phases) always come from base_path
    # (model_a), so alpha=0 doesn't match unmodified model_b.
    _HIDDEN = {"pretrained.decoder._hidden_state", "pretrained.encoder._hidden_state"}
    bufs_a = {k: v.clone() for k, v in model_a.named_buffers() if k not in _HIDDEN}
    bufs_b = {k: v.clone() for k, v in model_b.named_buffers() if k not in _HIDDEN}
    for name, buf in m_blend.named_buffers():
        if name in _HIDDEN:
            continue
        if name in bufs_a and name in bufs_b and bufs_a[name].shape == bufs_b[name].shape:
            buf.data.copy_(alpha * bufs_a[name] + (1.0 - alpha) * bufs_b[name])
        elif name in bufs_a and fallback == "a":
            buf.data.copy_(bufs_a[name])
        elif name in bufs_b and fallback == "b":
            buf.data.copy_(bufs_b[name])

    return m_blend


def alpha_sweep(
    model_a,
    model_b,
    alphas: list[float],
    base_path: str,
    fallback: str = "a",
) -> list[tuple[float, object]]:
    """Return [(alpha, blended_model), ...] for each alpha in `alphas`."""
    return [(a, blend(model_a, model_b, a, base_path, fallback)) for a in alphas]
