"""Feature extraction: A-weighted loudness and F0 (via torchcrepe)."""
import numpy as np
import torch
import torchaudio
import torchcrepe

SAMPLE_RATE = 16000
HOP_LENGTH = 64          # 250 Hz frame rate at 16 kHz
A_WEIGHT_DB_REF = -20.0  # floor for log loudness


def _a_weighting_weights(fft_size: int, sr: int) -> torch.Tensor:
    """Return A-weighting magnitude weights for FFT bins."""
    freqs = torch.fft.rfftfreq(fft_size, d=1.0 / sr)
    freqs = freqs.clamp(min=1e-6)
    f2 = freqs ** 2
    f4 = f2 ** 2
    numerator = (12194.0 ** 2) * f4
    denominator = (
        (f2 + 20.6 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    ra = numerator / denominator
    # Normalize so that ra(1000 Hz) ≈ 1
    f1k = torch.tensor(1000.0)
    f2_1k = f1k ** 2
    ra_1k = (12194.0 ** 2) * f2_1k ** 2 / (
        (f2_1k + 20.6 ** 2)
        * ((f2_1k + 107.7 ** 2) * (f2_1k + 737.9 ** 2)) ** 0.5
        * (f2_1k + 12194.0 ** 2)
    )
    return ra / ra_1k


def extract_loudness(
    audio: torch.Tensor,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    n_fft: int = 2048,
) -> torch.Tensor:
    """
    Compute per-frame A-weighted log loudness.

    Args:
        audio: (T,) waveform at ``sr``
        sr: sample rate
        hop_length: STFT hop in samples
        n_fft: FFT size

    Returns:
        loudness: (N_frames,) tensor in dB, normalised to roughly [-1, 1]
    """
    # Power spectrum via STFT
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=audio.device),
        return_complex=True,
    )  # (n_fft//2+1, N_frames)

    power = stft.abs() ** 2  # (bins, frames)
    weights = _a_weighting_weights(n_fft, sr).to(audio.device)  # (bins,)
    a_power = (power * weights.unsqueeze(1)).mean(dim=0)  # (frames,)
    loudness_db = 10.0 * torch.log10(a_power.clamp(min=1e-10))
    # Normalise so silence ≈ -1 and loud ≈ 0
    loudness = (loudness_db - A_WEIGHT_DB_REF) / (-A_WEIGHT_DB_REF)
    return loudness  # (N_frames,)


def extract_f0(
    audio: torch.Tensor,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    fmin: float = 32.7,   # C1
    fmax: float = 2093.0, # C7
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate per-frame F0 using torchcrepe.

    Returns:
        f0_hz:  (N_frames,) fundamental frequency in Hz (0 for unvoiced)
        voiced: (N_frames,) boolean mask
    """
    audio_np = audio.cpu().numpy()
    # torchcrepe expects (1, T) float32
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).float()

    f0, periodicity = torchcrepe.predict(
        audio_t,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model="tiny",
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )  # both (1, N_frames)

    f0 = f0.squeeze(0)           # (N_frames,)
    periodicity = periodicity.squeeze(0)

    voiced = periodicity > 0.21  # standard torchcrepe threshold
    f0_voiced = f0 * voiced.float()
    return f0_voiced, voiced
