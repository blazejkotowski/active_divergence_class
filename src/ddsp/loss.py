"""Multi-scale spectral loss (Engel et al. 2020, §4.2.1)."""
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
