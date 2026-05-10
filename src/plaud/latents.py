"""
Latent trajectory generators for PLAUD inference.

PLAUD's decoder expects a latent trajectory of shape (1, T, 4) where:
  T   = number of frames  (1 frame = 32 audio samples at 48 kHz = 1/1500 s)
  4   = latent dimensionality (discovered via decode_params buffer)

All generators return a tensor ready to pass to decode().
"""

import math
import torch

LATENT_DIM: int = 4
HOP: int = 32
SR: int = 48000
FRAMES_PER_SEC: float = SR / HOP  # 1500.0


def lfo_latent(
    duration_s: float,
    freqs: list[float] | None = None,
    global_freq_mult: float = 1.0,
    sr: int = SR,
    hop: int = HOP,
) -> torch.Tensor:
    """Multi-channel LFO bank → latent trajectory (1, T, 4).

    Each of the 4 latent channels is a sine wave with an independent
    frequency. Good default: [1.0, 2.0, 0.5, 0.3] Hz.
    """
    if freqs is None:
        freqs = [1.0, 2.0, 0.5, 0.3]
    freqs = list(freqs)[:LATENT_DIM]
    T = int(duration_s * sr / hop)
    t = torch.linspace(0.0, duration_s, T)
    channels = []
    for f in freqs:
        channels.append(torch.sin(2.0 * math.pi * f * global_freq_mult * t))
    while len(channels) < LATENT_DIM:
        channels.append(torch.zeros(T))
    z = torch.stack(channels, dim=-1)        # (T, 4)
    return z.unsqueeze(0)                    # (1, T, 4)


def constant_latent(
    duration_s: float,
    values: list[float] | None = None,
    sr: int = SR,
    hop: int = HOP,
) -> torch.Tensor:
    """Constant-vector latent trajectory (1, T, 4).

    Each channel is fixed at its given value. Useful for isolating the
    effect of bending from the effect of latent dynamics.
    """
    if values is None:
        values = [0.0, 0.0, 0.0, 0.0]
    values = list(values)[:LATENT_DIM]
    T = int(duration_s * sr / hop)
    channels = [torch.full((T,), float(v)) for v in values]
    while len(channels) < LATENT_DIM:
        channels.append(torch.zeros(T))
    return torch.stack(channels, dim=-1).unsqueeze(0)


def random_walk_latent(
    duration_s: float,
    step_size: float = 0.05,
    sr: int = SR,
    hop: int = HOP,
    seed: int = 42,
) -> torch.Tensor:
    """Random-walk latent trajectory (1, T, 4).

    Each channel performs a cumulative random walk, normalized to [-1, 1].
    Provides continuous but unpredictable variation — useful for stress-testing
    bending effects across a wide range of latent positions.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    T = int(duration_s * sr / hop)
    steps = torch.randn(T, LATENT_DIM, generator=gen) * step_size
    z = steps.cumsum(dim=0)
    z = z / (z.abs().amax() + 1e-8)
    return z.unsqueeze(0)
