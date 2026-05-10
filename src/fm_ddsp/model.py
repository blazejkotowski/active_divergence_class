"""
DDX7-style differentiable FM synthesizer for instrument resynthesis.

Two variants:
  DDSPFMModel      — fixed frequency ratios (DDX7 constraint, best training stability)
  DDSPFMModelFlex  — learnable frequency ratios (for §3.3 inharmonicity experiments)
"""

import torch
import torch.nn as nn

from .decoder import FMDecoder
from .synth import FMOperatorBank, VIOLIN_RATIOS, VIOLIN_ROUTING, VIOLIN_CARRIERS, VIOLIN_I_MAX
from src.ddsp.synth import FilteredNoise, Reverb

SAMPLE_RATE = 16000
HOP_LENGTH  = 64


class DDSPFMModel(nn.Module):
    """
    FM resynthesis model with fixed frequency ratios (violin "STRINGS 1" DX7 patch).

    Decoder: MLP+GRU → sigmoid envelopes for each of 6 FM operators + noise magnitudes.
    Synth:   FMOperatorBank (fixed ratios) + FilteredNoise (65 bands) summed before reverb.
    Reverb:  learnable IR (shared across batch, as in DDSP original).

    Args:
        hidden_size:    GRU/MLP hidden dimension
        n_mlp_layers:   depth of input/output MLPs
        n_noise_bands:  number of noise filter bands (65 matches DDSP baseline)
        I_max:          maximum FM modulation index
        reverb_length:  learnable IR length in samples
        sample_rate:    audio sample rate
        hop_length:     frame → sample stride
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_mlp_layers: int = 3,
        n_noise_bands: int = 65,
        I_max: float = VIOLIN_I_MAX,
        reverb_length: int = 48000,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        ratios: list[float] = VIOLIN_RATIOS,
        routing: list[tuple[int, int]] = VIOLIN_ROUTING,
        carriers: list[int] = VIOLIN_CARRIERS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length  = hop_length
        self.n_ops       = len(ratios)

        self.decoder      = FMDecoder(n_ops=self.n_ops, n_noise_bands=n_noise_bands,
                                      hidden_size=hidden_size, n_mlp_layers=n_mlp_layers)
        self.fm_synth     = FMOperatorBank(ratios=ratios, routing=routing, carriers=carriers,
                                           I_max=I_max, sample_rate=sample_rate, hop_length=hop_length)
        self.noise_synth  = FilteredNoise(n_bands=n_noise_bands, hop_length=hop_length)
        self.reverb       = Reverb(ir_length=reverb_length, sample_rate=sample_rate)

    def forward(
        self,
        f0_hz: torch.Tensor,     # (B, N_frames)
        loudness: torch.Tensor,  # (B, N_frames)
    ) -> dict[str, torch.Tensor]:
        envelopes, noise_mags = self.decoder(f0_hz, loudness)  # (B, N_ops, N_frames), (B, N, K)
        fm_dry  = self.fm_synth(f0_hz, envelopes)              # (B, T)
        noise   = self.noise_synth(noise_mags)                 # (B, T)
        T = min(fm_dry.shape[-1], noise.shape[-1])
        dry      = fm_dry[..., :T] + noise[..., :T]
        audio    = self.reverb(dry)
        fm_audio = self.reverb(fm_dry[..., :T])               # FM only, no noise
        return {"audio": audio, "dry_audio": dry, "operator_envelopes": envelopes,
                "fm_audio": fm_audio}

    @torch.no_grad()
    def synthesise(self, f0_hz, loudness):
        return self.forward(f0_hz, loudness)["audio"]


class DDSPFMModelFlex(DDSPFMModel):
    """
    FM model with learnable frequency ratios — for inharmonicity experiments.

    Inherits DDSPFMModel but replaces the fixed ratio buffer with an nn.Parameter.
    The ratios start at violin "STRINGS 1" values and drift during fine-tuning.

    When ratios deviate from integers (e.g. 3.0 → 3.47), the FM sidebands
    land at non-harmonic positions → genuine inharmonicity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Lift fixed_ratios from buffer to learnable parameter.
        # Initialise in softplus-inverse space so ratios start near violin values.
        # softplus(x) + 0.5 = ratio  →  x = log(exp(ratio - 0.5) - 1)
        init_ratios = self.fm_synth.fixed_ratios.clone()
        init_raw = (init_ratios - 0.5).clamp(min=1e-4).expm1().clamp(min=1e-4).log()
        del self.fm_synth.fixed_ratios
        self.fm_synth.fixed_ratios = None
        self.learned_ratios = nn.Parameter(init_raw)

    def forward(self, f0_hz, loudness):
        # softplus(beta=1) + 0.5 keeps every ratio ≥ 0.5 (all operators stay audible).
        # Init formula in __init__ targets beta=1, so ratios start at violin values.
        r = torch.nn.functional.softplus(self.learned_ratios) + 0.5
        envelopes, noise_mags = self.decoder(f0_hz, loudness)
        fm_dry = self.fm_synth(f0_hz, envelopes, ratios=r)
        noise  = self.noise_synth(noise_mags)
        T = min(fm_dry.shape[-1], noise.shape[-1])
        dry      = fm_dry[..., :T] + noise[..., :T]
        audio    = self.reverb(dry)
        fm_audio = self.reverb(fm_dry[..., :T])               # FM only, no noise
        return {
            "audio": audio,
            "dry_audio": dry,
            "operator_envelopes": envelopes,
            "ratios": r,
            "fm_audio": fm_audio,
        }
