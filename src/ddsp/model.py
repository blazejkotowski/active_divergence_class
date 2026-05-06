"""Full DDSP autoencoder: encoder (features) + decoder + synthesizers."""
import torch
import torch.nn as nn

from .decoder import DDSPDecoder
from .synth import HarmonicOscillator, FilteredNoise, Reverb

SAMPLE_RATE = 16000
HOP_LENGTH = 64


class DDSPAutoencoder(nn.Module):
    """
    Supervised DDSP autoencoder (Engel et al. 2020, §4.1).

    The encoder is hand-crafted: loudness (A-weighted log) + F0 from CREPE.
    The decoder is a learned MLP+GRU that maps these to synth parameters.
    No Z encoder is used (minimal baseline).
    """

    def __init__(
        self,
        n_harmonics: int = 100,
        n_noise_bands: int = 65,
        hidden_size: int = 512,
        n_mlp_layers: int = 3,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        reverb_ir_length: int = 64000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.decoder = DDSPDecoder(
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_bands,
            hidden_size=hidden_size,
            n_mlp_layers=n_mlp_layers,
        )
        self.harmonic_osc = HarmonicOscillator(
            n_harmonics=n_harmonics,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.filtered_noise = FilteredNoise(
            n_bands=n_noise_bands,
            hop_length=hop_length,
        )
        self.reverb = Reverb(ir_length=reverb_ir_length, sample_rate=sample_rate)


    def forward(
        self,
        f0_hz: torch.Tensor,    # (B, N_frames)
        loudness: torch.Tensor,  # (B, N_frames)
    ) -> dict[str, torch.Tensor]:
        controls = self.decoder(f0_hz, loudness)

        harmonic_audio = self.harmonic_osc(
            f0_hz=f0_hz,
            amplitudes=controls["harmonic_dist"],
            global_amp=controls["global_amp"],
        )
        noise_audio = self.filtered_noise(controls["noise_mag"])

        dry = harmonic_audio + noise_audio   # (B, T)
        audio = self.reverb(dry)             # (B, T) — room acoustics applied

        return {
            "audio": audio,
            "dry_audio": dry,
            "harmonic_audio": harmonic_audio,
            "noise_audio": noise_audio,
            **controls,
        }

    @torch.no_grad()
    def synthesise(
        self,
        f0_hz: torch.Tensor,
        loudness: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience inference method (no grad). Returns (B, T) audio."""
        out = self.forward(f0_hz, loudness)
        return out["audio"]
