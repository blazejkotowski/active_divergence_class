"""GRU+MLP decoder: (f0, loudness) → per-operator FM envelopes + noise magnitudes."""

import torch
import torch.nn as nn


class FMDecoder(nn.Module):
    """
    Maps per-frame (f0_hz, loudness) to FM operator envelopes and noise magnitudes.

    Architecture mirrors the DDSP decoder (MLP → GRU → MLP → heads) with two output
    heads: one for FM operator envelopes, one for filtered-noise frequency magnitudes.

    Args:
        n_ops:        number of FM operators (one envelope per operator)
        n_noise_bands: number of noise filter bands (matches FilteredNoise)
        hidden_size:  GRU and MLP hidden dimension
        n_mlp_layers: depth of input and output MLPs
    """

    def __init__(
        self,
        n_ops: int = 6,
        n_noise_bands: int = 65,
        hidden_size: int = 256,
        n_mlp_layers: int = 3,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.n_noise_bands = n_noise_bands

        layers_in = []
        d = 2
        for _ in range(n_mlp_layers):
            layers_in += [nn.Linear(d, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()]
            d = hidden_size
        self.input_mlp = nn.Sequential(*layers_in)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        layers_out = []
        for _ in range(n_mlp_layers):
            layers_out += [nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()]
        self.output_mlp = nn.Sequential(*layers_out)

        self.env_head   = nn.Linear(hidden_size, n_ops)
        self.noise_head = nn.Linear(hidden_size, n_noise_bands)

    def forward(
        self,
        f0_hz: torch.Tensor,     # (B, N_frames)
        loudness: torch.Tensor,  # (B, N_frames)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: envelopes (B, N_ops, N_frames), noise_mags (B, N_frames, N_bands)
        f0_norm = (f0_hz.clamp(min=1.0).log() - 3.4812) / (7.6474 - 3.4812)
        x = torch.stack([f0_norm, loudness], dim=-1)   # (B, N, 2)

        x = self.input_mlp(x)      # (B, N, H)
        x, _ = self.gru(x)         # (B, N, H)
        x = self.output_mlp(x)     # (B, N, H)

        envelopes  = torch.sigmoid(self.env_head(x))    # (B, N, N_ops)
        noise_mags = torch.sigmoid(self.noise_head(x))  # (B, N, N_bands)
        return envelopes.permute(0, 2, 1), noise_mags   # (B, N_ops, N_frames), (B, N_frames, N_bands)
