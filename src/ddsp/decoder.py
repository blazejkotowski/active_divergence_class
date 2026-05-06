"""MLP+GRU decoder: (f0, loudness) → synth controls."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Non-negative output activation from Engel et al. (appendix)."""
    return 2.0 * torch.sigmoid(x) ** 2.3 + 1e-7


class DDSPDecoder(nn.Module):
    """
    Maps per-frame (f0_hz, loudness) to harmonic + noise synth parameters.

    Architecture: MLP(input) → GRU → MLP(output heads)
    following the DDSP paper conventions (fully connected + single recurrent layer).
    """

    def __init__(
        self,
        n_harmonics: int = 100,
        n_noise_bands: int = 65,
        hidden_size: int = 512,
        n_mlp_layers: int = 3,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.hidden_size = hidden_size

        # Input: f0 (normalised) + loudness → 2 scalars per frame
        input_dim = 2

        # Input MLP: project scalars into hidden space
        mlp_in = []
        in_d = input_dim
        for _ in range(n_mlp_layers):
            mlp_in += [nn.Linear(in_d, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()]
            in_d = hidden_size
        self.input_mlp = nn.Sequential(*mlp_in)

        # Recurrent layer for temporal context
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )

        # Output MLP
        mlp_out = []
        for _ in range(n_mlp_layers):
            mlp_out += [nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()]
        self.output_mlp = nn.Sequential(*mlp_out)

        # Output heads
        # Global amplitude: scalar per frame
        self.head_amp = nn.Linear(hidden_size, 1)
        # Per-harmonic distribution c_k (sum-to-one via softmax)
        self.head_harmonics = nn.Linear(hidden_size, n_harmonics)
        # Noise band magnitudes
        self.head_noise = nn.Linear(hidden_size, n_noise_bands)

    def forward(
        self,
        f0_hz: torch.Tensor,    # (B, N_frames)
        loudness: torch.Tensor,  # (B, N_frames)
    ) -> dict[str, torch.Tensor]:
        B, N = f0_hz.shape

        # Normalise f0 to [0, 1] range (roughly, based on violin range 196-3136 Hz)
        f0_norm = (torch.log(f0_hz.clamp(min=1e-3)) - torch.log(torch.tensor(32.7))) / (
            torch.log(torch.tensor(2093.0)) - torch.log(torch.tensor(32.7))
        )
        f0_norm = f0_norm.unsqueeze(-1)   # (B, N, 1)
        loud = loudness.unsqueeze(-1)      # (B, N, 1)

        x = torch.cat([f0_norm, loud], dim=-1)  # (B, N, 2)

        # MLP → GRU → MLP
        x = self.input_mlp(x)          # (B, N, H)
        x, _ = self.gru(x)             # (B, N, H)
        x = self.output_mlp(x)         # (B, N, H)

        # Output heads with non-negative activations
        global_amp = modified_sigmoid(self.head_amp(x)).squeeze(-1)        # (B, N)
        harmonic_dist = torch.softmax(self.head_harmonics(x), dim=-1)      # (B, N, K)
        noise_mag = modified_sigmoid(self.head_noise(x))                   # (B, N, M)

        return {
            "global_amp": global_amp,
            "harmonic_dist": harmonic_dist,
            "noise_mag": noise_mag,
        }
