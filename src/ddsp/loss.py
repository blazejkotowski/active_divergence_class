"""Multi-scale spectral loss (Engel et al. 2020, §4.2.1) + inharmonicity loss."""
import torch
import torch.nn as nn


class MultiScaleSpectralLoss(nn.Module):
    """
    L_i = ||S_i - S_hat_i||_1 + alpha * ||log S_i - log S_hat_i||_1
    summed over FFT sizes, each with 75% overlap.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64),
        alpha: float = 1.0,
        overlap: float = 0.75,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.alpha = alpha
        self.overlap = overlap

    def forward(self, audio: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(1, device=audio.device)
        for fft_size in self.fft_sizes:
            hop = int(fft_size * (1.0 - self.overlap))
            hop = max(hop, 1)
            win = torch.hann_window(fft_size, device=audio.device)

            def spec(x):
                return torch.stft(
                    x,
                    n_fft=fft_size,
                    hop_length=hop,
                    win_length=fft_size,
                    window=win,
                    return_complex=True,
                ).abs()

            S = spec(target)
            S_hat = spec(audio)

            lin_loss = (S - S_hat).abs().mean()
            log_loss = (S.clamp(min=1e-7).log() - S_hat.clamp(min=1e-7).log()).abs().mean()
            total = total + lin_loss + self.alpha * log_loss

        return total


class InharmonicityLoss(nn.Module):
    """
    Inharmonicity loss for DDSP training-time divergence (Section 3.3).

    Operates directly on the decoder's harmonic amplitude distribution —
    no FFT needed, cheap on CPU.

    **Definition** — we define inharmonicity as entropy of the harmonic distribution:

        H(p) = - sum_k  p_k * log(p_k + eps)

    where p_k is the softmax weight of harmonic k, averaged over batch (B) and
    frames (N).  A concentrated distribution (most energy in harmonics 1-3) has
    LOW entropy; a spread distribution (energy across all 100 harmonics) has HIGH
    entropy.  Bell-like / metallic / inharmonic timbres tend to have higher spectral
    spread, so maximising H pushes the model toward those timbres.

    The loss returned is **negative entropy** (so minimising it maximises spread):

        L_inh = -mean_{B,N}[ sum_k p_k * log(p_k + eps) ]

    Use it as:  total_loss = spectral_loss + weight * inh_loss

    Args:
        eps: numerical floor to prevent log(0)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, harmonic_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            harmonic_dist: (B, N_frames, K) — softmax harmonic amplitude weights
                           (output of DDSPDecoder, already normalised to sum to 1)
        Returns:
            scalar negative entropy (minimise to spread harmonics)
        """
        # Shannon entropy per frame: -sum_k p_k * log(p_k)
        entropy = -(harmonic_dist * (harmonic_dist + self.eps).log()).sum(dim=-1)
        # Mean over batch and frames; negate to get a loss (we want to maximise entropy)
        return -entropy.mean()
