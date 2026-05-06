"""DDSP synthesis modules: harmonic oscillator + filtered noise."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
HOP_LENGTH = 64


class HarmonicOscillator(nn.Module):
    """
    Additive sinusoidal synthesizer (Engel et al. 2020, §3.2).

    Inputs (all frame-rate, interpolated to audio-rate internally):
        f0_hz:        (B, N_frames)  fundamental frequency in Hz
        amplitudes:   (B, N_frames, N_harmonics)  per-harmonic amplitudes
        global_amp:   (B, N_frames)  overall amplitude envelope

    Output:
        audio: (B, T)  synthesised waveform
    """

    def __init__(
        self,
        n_harmonics: int = 100,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(
        self,
        f0_hz: torch.Tensor,
        amplitudes: torch.Tensor,
        global_amp: torch.Tensor,
    ) -> torch.Tensor:
        B, N = f0_hz.shape
        T = N * self.hop_length

        # Harmonic multipliers: [1, 2, ..., N_harmonics]
        k = torch.arange(1, self.n_harmonics + 1, device=f0_hz.device).float()

        # Upsample frame-rate controls to audio rate via linear interpolation
        # f0: (B, 1, N) → (B, 1, T)
        f0_up = F.interpolate(
            f0_hz.unsqueeze(1), size=T, mode="linear", align_corners=True
        ).squeeze(1)  # (B, T)

        # amp per harmonic: (B, N, K) → (B, K, N) → (B, K, T) → (B, T, K)
        amp_up = F.interpolate(
            amplitudes.permute(0, 2, 1),  # (B, K, N)
            size=T,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)  # (B, T, K)

        global_amp_up = F.interpolate(
            global_amp.unsqueeze(1), size=T, mode="linear", align_corners=True
        ).squeeze(1)  # (B, T)

        # Anti-aliasing: zero out harmonics above Nyquist
        # f0_up: (B, T), k: (K,)  → (B, T, K)
        harmonic_freqs = f0_up.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0)
        above_nyquist = harmonic_freqs >= (self.sample_rate / 2.0)
        amp_up = amp_up * (~above_nyquist).float()

        # Integrate instantaneous frequency → phase
        # freq in cycles/sample = freq_hz / sr
        harmonic_freqs_cps = harmonic_freqs / self.sample_rate  # (B, T, K)
        # Cumulative sum along time axis (Euler integration)
        phase = 2.0 * torch.pi * torch.cumsum(harmonic_freqs_cps, dim=1)  # (B, T, K)

        # Synthesise: sum of weighted sinusoids
        # A_k(n) = global_amp(n) * amplitude_k(n)
        weighted_amp = global_amp_up.unsqueeze(-1) * amp_up  # (B, T, K)
        audio = (weighted_amp * torch.sin(phase)).sum(dim=-1)  # (B, T)
        return audio


class FilteredNoise(nn.Module):
    """
    Filtered noise synthesizer via time-varying FIR filter (Engel et al. 2020, §3.4-3.5).

    Inputs:
        noise_magnitudes: (B, N_frames, N_bands) — frequency-domain magnitudes

    Output:
        audio: (B, T) filtered white noise
    """

    def __init__(
        self,
        n_bands: int = 65,
        hop_length: int = HOP_LENGTH,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.hop_length = hop_length
        # IR length derived from n_bands: n_fft = (n_bands-1)*2
        self.n_fft = (n_bands - 1) * 2

    def forward(self, noise_magnitudes: torch.Tensor) -> torch.Tensor:
        B, N, K = noise_magnitudes.shape
        T = N * self.hop_length
        device = noise_magnitudes.device
        n_fft = self.n_fft  # = (K-1)*2

        # Generate white noise
        noise = torch.rand(B, T, device=device) * 2.0 - 1.0

        # Build per-frame impulse responses from magnitude spectrum
        # noise_magnitudes: (B, N, K) → (B*N, K)
        mag = noise_magnitudes.reshape(B * N, K).to(torch.complex64)

        # IDFT → zero-phase IR of length n_fft
        ir = torch.fft.irfft(mag, n=n_fft)  # (B*N, n_fft)

        # Apply Hann window of size n_fft (zero-phase form, then causal shift)
        win = torch.hann_window(n_fft, device=device)
        ir = ir * win  # window already the same length as ir

        # Causal shift: roll by half so energy is at the start
        half = n_fft // 2
        ir_causal = torch.roll(ir, half, dims=-1)

        # Apply per-frame FIR via overlap-add convolution
        # Reshape noise to frames: (B, N, hop_length)
        noise_frames = noise.reshape(B, N, self.hop_length)

        # For each frame, convolve noise chunk with its IR
        # Simple freq-domain frame-by-frame convolution
        frame_len = self.hop_length
        fft_size = n_fft + frame_len  # linear convolution
        noise_fft = torch.fft.rfft(noise_frames.reshape(B * N, frame_len), n=fft_size)
        ir_fft = torch.fft.rfft(ir_causal, n=fft_size)
        conv = torch.fft.irfft(noise_fft * ir_fft, n=fft_size)  # (B*N, fft_size)

        # Overlap-add: each frame contributes hop_length samples + tail
        conv = conv.reshape(B, N, fft_size)
        out = torch.zeros(B, T + n_fft, device=device)
        for i in range(N):
            start = i * self.hop_length
            out[:, start : start + fft_size] += conv[:, i, :]

        return out[:, :T]


class Reverb(nn.Module):
    """
    Learned fixed-IR reverb via frequency-domain convolution (Engel et al. 2020, §3.6).

    A single nn.Parameter IR is shared across all examples in a batch.
    This works because the violin recordings share the same room.
    IR length = sample_rate * 4 = 64000 at 16 kHz.
    """

    def __init__(self, ir_length: int = 64000, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.ir_length = ir_length
        # Initialise with a decaying exponential so it sounds like a room from step 0
        t = torch.linspace(0, 1, ir_length)
        init_ir = torch.exp(-10.0 * t) * torch.randn(ir_length) * 0.1
        self.ir = nn.Parameter(init_ir)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T)
        T = audio.shape[-1]
        fft_size = T + self.ir_length  # linear (non-circular) convolution length
        audio_fft = torch.fft.rfft(audio, n=fft_size)              # (B, F)
        ir_fft    = torch.fft.rfft(self.ir.unsqueeze(0), n=fft_size)  # (1, F)
        wet = torch.fft.irfft(audio_fft * ir_fft, n=fft_size)     # (B, fft_size)
        return wet[..., :T]
