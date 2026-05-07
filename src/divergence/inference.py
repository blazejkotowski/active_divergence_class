"""
src/divergence/inference.py
Inference-time active divergence utilities for the DDSP violin model.

All functions assume CPU execution and frozen model weights.
Designed for use in notebooks/02_inference_divergence.ipynb.
"""
import sys
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import Audio, display

ROOT = pathlib.Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ddsp.model import DDSPAutoencoder


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cpu") -> DDSPAutoencoder:
    """Load a trained DDSPAutoencoder checkpoint with weights frozen."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DDSPAutoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Duration → frame count
# ─────────────────────────────────────────────────────────────────────────────

def n_frames(duration_s: float, sr: int = 16000, hop: int = 64) -> int:
    """Convert a duration in seconds to a frame count."""
    return int(duration_s * sr / hop)


# ─────────────────────────────────────────────────────────────────────────────
# OOD input trajectory generators
# All return (1, N) shaped tensors ready for model(f0, loudness).
# ─────────────────────────────────────────────────────────────────────────────

def constant(f0_hz: float, loudness_db: float, nf: int):
    """Flat pitch and loudness — simplest possible trajectory."""
    return torch.full((1, nf), f0_hz), torch.full((1, nf), loudness_db)


def pitch_glide(f0_start: float, f0_end: float, nf: int, log: bool = True):
    """Smooth pitch glide. log=True gives perceptually linear movement."""
    if log:
        f0 = torch.logspace(np.log10(max(f0_start, 0.1)),
                            np.log10(max(f0_end,   0.1)), nf)
    else:
        f0 = torch.linspace(f0_start, f0_end, nf)
    return f0.unsqueeze(0)


def microtonal_glide(f0_center: float, semitone_range: float, nf: int,
                     cycles: float = 4.0):
    """Sinusoidal microtonal pitch oscillation around a centre frequency."""
    t = torch.linspace(0, 2 * np.pi * cycles, nf)
    f0 = f0_center * 2 ** (semitone_range / 2 * torch.sin(t) / 12.0)
    return f0.unsqueeze(0)


def fast_am(nf: int, rate_hz: float = 8.0, depth_db: float = 30.0,
            base_db: float = -30.0, sr: int = 16000, hop: int = 64):
    """Sinusoidal amplitude modulation of the loudness control."""
    t = torch.arange(nf).float() * hop / sr
    mod = 0.5 + 0.5 * torch.sin(2 * np.pi * rate_hz * t)
    return (base_db + depth_db * mod).unsqueeze(0)


def square_gate(nf: int, rate_hz: float = 2.0,
                on_db: float = -20.0, off_db: float = -80.0,
                sr: int = 16000, hop: int = 64):
    """Binary on/off loudness gating — hard cut envelopes."""
    t = torch.arange(nf).float() * hop / sr
    gate = (torch.sin(2 * np.pi * rate_hz * t) > 0).float()
    return (on_db * gate + off_db * (1 - gate)).unsqueeze(0)


def random_walk(nf: int, start: float, step_std: float,
                lo: float, hi: float):
    """Brownian motion trajectory clamped to [lo, hi]."""
    return (torch.cumsum(torch.randn(nf) * step_std, dim=0)
            .add_(start).clamp_(lo, hi).unsqueeze(0))


def periodic_pitch(f0_center: float, nf: int,
                   rate_hz: float = 0.5, semitone_range: float = 24.0,
                   sr: int = 16000, hop: int = 64):
    """Structured periodic pitch oscillation — wide vibrato / LFO sweep."""
    t = torch.arange(nf).float() * hop / sr
    f0 = f0_center * 2 ** (semitone_range / 2
                           * torch.sin(2 * np.pi * rate_hz * t) / 12)
    return f0.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder interception
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_controls(model: DDSPAutoencoder,
                 f0_hz: torch.Tensor,
                 loudness: torch.Tensor) -> dict:
    """Run only the decoder; return raw synth-parameter dict."""
    return model.decoder(f0_hz, loudness)


# ─────────────────────────────────────────────────────────────────────────────
# Bent harmonic oscillator
# ─────────────────────────────────────────────────────────────────────────────

def _up(x: torch.Tensor, T: int) -> torch.Tensor:
    """Linearly interpolate (B, N) or (B, N, K) from N frames to T samples."""
    if x.dim() == 2:
        return F.interpolate(x.unsqueeze(1), T, mode="linear",
                             align_corners=True).squeeze(1)
    B, N, K = x.shape
    return (F.interpolate(x.permute(0, 2, 1), T, mode="linear",
                          align_corners=True).permute(0, 2, 1))


def harmonic_oscillator_bent(
    f0_hz:      torch.Tensor,   # (B, N)
    amplitudes: torch.Tensor,   # (B, N, K)
    global_amp: torch.Tensor,   # (B, N)
    *,
    waveform_fn=None,           # callable(phase: (B,T,K)) -> (B,T,K); default sin
    fm_depth:      float = 0.0,  # FM modulation index in radians
    fm_ratio:      float = 1.0,  # modulator freq = f0 × fm_ratio per harmonic
    inharmonicity: float = 0.0,  # B coefficient: f_k = k·f0·√(1 + B·k²); 0 = harmonic
    h_n:           int   = None, # limit inharmonicity to first h_n harmonics
    sample_rate: int  = 16000,
    hop_length:  int  = 64,
) -> torch.Tensor:
    """
    Additive synthesiser with three bending hooks:

    1. `waveform_fn`: replaces sin(phase).
    2. `fm_depth`: per-harmonic FM synthesis.
    3. `inharmonicity`: stretches harmonic frequencies via the piano inharmonicity
       formula  f_k = k·f0·√(1 + B·k²).  h_n limits which harmonics are affected.

    Follows Engel et al. (2020, §3.2) but with the synthesis function exposed.
    """
    B, N = f0_hz.shape
    T = N * hop_length
    K = amplitudes.shape[-1]
    k_idx = torch.arange(1, K + 1, device=f0_hz.device).float()

    f0_up  = _up(f0_hz,      T)   # (B, T)
    amp_up = _up(amplitudes,  T)   # (B, T, K)
    g_up   = _up(global_amp,  T)   # (B, T)

    # Inharmonicity: stretch harmonic frequencies f_k = k·f0·√(1 + B·k²)
    if inharmonicity != 0.0:
        stretch = torch.sqrt((1.0 + inharmonicity * k_idx ** 2).clamp(min=1e-4))
        if h_n is not None and h_n < K:
            scoped = torch.ones(K, device=f0_hz.device)
            scoped[:h_n] = stretch[:h_n]
            stretch = scoped
        hfreqs = f0_up.unsqueeze(-1) * k_idx * stretch   # (B, T, K) Hz
    else:
        hfreqs = f0_up.unsqueeze(-1) * k_idx             # (B, T, K) Hz

    # Anti-aliasing: silence harmonics above Nyquist
    amp_up = amp_up * (hfreqs < sample_rate / 2).float()

    # Instantaneous phase via Euler integration of frequency
    phase = 2 * np.pi * torch.cumsum(hfreqs / sample_rate, dim=1)  # (B, T, K)

    # FM bending: add a sinusoidal phase modulation from a co-generated modulator
    if fm_depth != 0.0:
        mod_phase = 2 * np.pi * torch.cumsum(
            f0_up.unsqueeze(-1) * fm_ratio / sample_rate, dim=1)
        phase = phase + fm_depth * torch.sin(mod_phase)

    # Base waveform hook — the divergence point
    waveform_fn = waveform_fn if waveform_fn is not None else torch.sin
    wave = waveform_fn(phase)   # (B, T, K)

    return (g_up.unsqueeze(-1) * amp_up * wave).sum(-1)   # (B, T)


WAVEFORM_NAMES = [
    "sin (original)", "tanh", "abs (rectify)",
    "hard clip", "square", "triangle", "sawtooth", "wavefold",
]


def make_waveform_fn(name: str, amount: float = 2.0):
    """
    Parametric waveform factory for interactive demos.

    `amount` controls the strength of the distortion:
      tanh      — pre-gain (1 = mild, 10 = near-square)
      hard clip — inverse threshold (1 = identity, 10 = extreme)
      wavefold  — fold gain (1 = mild, 5 = dense)
      others    — ignored (fixed shape)
    """
    if name == "sin (original)":
        return None
    elif name == "tanh":
        return lambda p: torch.tanh(amount * torch.sin(p))
    elif name == "abs (rectify)":
        return lambda p: torch.abs(torch.sin(p))
    elif name == "hard clip":
        t = max(1.0 / max(amount, 0.1), 0.05)
        return lambda p: torch.clamp(torch.sin(p), -t, t) / max(t, 1e-6)
    elif name == "square":
        return lambda p: torch.sign(torch.sin(p)).clamp(-1, 1)
    elif name == "triangle":
        return lambda p: (2 / np.pi) * torch.arcsin(torch.sin(p).clamp(-1, 1))
    elif name == "sawtooth":
        return lambda p: torch.remainder(p, 2 * np.pi) / np.pi - 1.0
    elif name == "wavefold":
        return lambda p: (
            torch.abs(torch.remainder(amount * torch.sin(p) + 1, 2) - 1) * 2 - 1
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic amplitude manipulation
# ─────────────────────────────────────────────────────────────────────────────

def mask_harmonics(harmonic_dist: torch.Tensor,
                   mode: str = "all",
                   k: int = 10,
                   rand_keep: float = 0.5,
                   n_harmonics=None) -> torch.Tensor:
    """
    Zero out subsets of harmonics.
    If n_harmonics is set, the mask is applied only within the first
    n_harmonics; higher harmonics are left unchanged.

    mode  description
    ----  -----------
    all         no masking (passthrough)
    odd         keep only odd-numbered harmonics (1, 3, 5, …)
    even        keep only even-numbered harmonics (2, 4, 6, …)
    first_k     keep only the lowest k harmonics
    above_k     keep only harmonics above index k
    random      randomly keep each harmonic with probability rand_keep
    """
    if n_harmonics is not None and n_harmonics < harmonic_dist.shape[-1]:
        result = harmonic_dist.clone()
        result[..., :n_harmonics] = mask_harmonics(
            harmonic_dist[..., :n_harmonics], mode=mode, k=k, rand_keep=rand_keep)
        return result
    H = harmonic_dist.shape[-1]
    idx = torch.arange(1, H + 1)
    if   mode == "all":     return harmonic_dist
    elif mode == "odd":     mask = (idx % 2 == 1).float()
    elif mode == "even":    mask = (idx % 2 == 0).float()
    elif mode == "first_k": mask = (idx <= k).float()
    elif mode == "above_k": mask = (idx > k).float()
    elif mode == "random":  mask = (torch.rand(H) < rand_keep).float()
    else:                   return harmonic_dist
    return harmonic_dist * mask.to(harmonic_dist.device)


# ─────────────────────────────────────────────────────────────────────────────
# Noise filter magnitude manipulation
# ─────────────────────────────────────────────────────────────────────────────

def bend_noise_mag(noise_mag: torch.Tensor,
                   mode: str = "original",
                   lo: int = 20,
                   hi: int = 50,
                   smooth_k: int = 7) -> torch.Tensor:
    """
    Transform noise-band magnitudes before FilteredNoise synthesis.

    mode           description
    ----           -----------
    original       passthrough
    invert         spectral inversion (flip magnitude profile)
    randomise      white noise (flat random magnitudes)
    low_pass       zero bands above `hi`
    high_pass      zero bands below `lo`
    notch          zero bands in [lo, hi]
    swap_frames    randomly permute frames (destroys temporal coherence)
    smooth         convolve band dimension with a box filter (blur spectrum)
    sharpen        high-boost: original + 3 × (original − smooth)
    """
    B, N, K = noise_mag.shape
    m = noise_mag.clone()
    if mode == "original":
        return m
    if mode == "invert":
        return m.amax(-1, keepdim=True) + 1e-7 - m
    if mode == "randomise":
        return torch.rand_like(m)
    if mode == "low_pass":
        mask = torch.zeros(K, device=m.device); mask[:hi] = 1.0
        return m * mask
    if mode == "high_pass":
        mask = torch.zeros(K, device=m.device); mask[lo:] = 1.0
        return m * mask
    if mode == "notch":
        mask = torch.ones(K, device=m.device); mask[lo:hi] = 0.0
        return m * mask
    if mode == "swap_frames":
        return m[:, torch.randperm(N), :]
    if mode in ("smooth", "sharpen"):
        pad = smooth_k // 2
        kernel = torch.ones(1, 1, smooth_k, device=m.device) / smooth_k
        flat = m.reshape(B * N, 1, K)
        sm = F.conv1d(flat, kernel, padding=pad)[:, 0, :K].reshape(B, N, K)
        return sm if mode == "smooth" else (m + 3.0 * (m - sm)).clamp(min=0)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Reverb IR manipulation
# ─────────────────────────────────────────────────────────────────────────────

def get_ir(model: DDSPAutoencoder) -> torch.Tensor:
    """Extract the learned reverb IR (shape [ir_length]) as a detached tensor."""
    return model.reverb.ir.detach().clone()


def bend_ir(ir: torch.Tensor, mode: str = "original",
            lo_hz: float = 300.0, hi_hz: float = 3000.0,
            sr: int = 16000) -> torch.Tensor:
    """
    Manipulate a reverb impulse response before convolution.

    mode              description
    ----              -----------
    original          passthrough
    invert            negate the IR (phase inversion)
    reverse           time-reverse the IR (reverse reverb)
    randomise_phase   keep magnitude spectrum, randomise phase (different room)
    low_pass          keep only energy below lo_hz
    high_pass         keep only energy above hi_hz
    notch             remove energy in [lo_hz, hi_hz]
    """
    ir = ir.clone()
    L = ir.shape[0]
    if mode == "original":   return ir
    if mode == "invert":     return -ir
    if mode == "reverse":    return ir.flip(0)
    if mode == "randomise_phase":
        spec  = torch.fft.rfft(ir)
        phase = (torch.rand(spec.shape) * 2 * np.pi).to(torch.float32)
        spec_bent = torch.abs(spec) * torch.exp(1j * phase.to(torch.complex64))
        return torch.fft.irfft(spec_bent, n=L)
    freqs = torch.fft.rfftfreq(L) * sr
    spec  = torch.fft.rfft(ir)
    if   mode == "low_pass":  spec[freqs > lo_hz] = 0.0
    elif mode == "high_pass": spec[freqs < hi_hz] = 0.0
    elif mode == "notch":     spec[(freqs > lo_hz) & (freqs < hi_hz)] = 0.0
    return torch.fft.irfft(spec, n=L)


def apply_ir(audio: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Convolve (B, T) audio with a 1-D IR via frequency-domain multiplication."""
    T = audio.shape[-1]
    fft_n = T + ir.shape[0]
    return torch.fft.irfft(
        torch.fft.rfft(audio, n=fft_n)
        * torch.fft.rfft(ir.unsqueeze(0), n=fft_n),
        n=fft_n,
    )[..., :T]


# ─────────────────────────────────────────────────────────────────────────────
# Full bent synthesis pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def synth_bent(
    model:    DDSPAutoencoder,
    f0_hz:    torch.Tensor,
    loudness: torch.Tensor,
    # --- Waveform bending ---
    wavefold:  float = 0.0,    # fold amount; 0 = sin (unchanged)
    fm_depth:  float = 0.0,
    fm_ratio:  float = 1.0,
    # --- Harmonic manipulation ---
    h_n:              int   = None,  # harmonics in scope; None = all 100
    h_inharmonicity:  float = 0.0,   # B coeff; 0 = perfect harmonics
    h_mask:           str   = "all",
    h_mask_k:         int   = 10,
    # --- Component levels ---
    harmonic_gain: float = 1.0,  # 0.0 to silence harmonics entirely
    noise_gain:    float = 1.0,  # 0.0 to silence noise entirely
    # --- Reverb ---
    ir_reverse: bool = False,    # True = reverse reverb
) -> dict:
    """
    End-to-end bent synthesis with all interception points.

    Decoder weights are frozen. Divergence is structural:
      1. wavefold / fm_depth replace the sinusoidal base waveform.
      2. h_inharmonicity stretches harmonic frequencies (within first h_n).
      3. h_mask zeros out harmonic subsets (within first h_n).
      4. harmonic_gain / noise_gain scale each component before mixing.
      5. ir_reverse time-reverses the learned reverb IR.

    Returns a dict: audio, dry, harmonic_audio, noise_audio,
                    harmonic_dist, noise_mag, global_amp.
    """
    controls = model.decoder(f0_hz, loudness)

    harm = controls["harmonic_dist"].clone()
    if h_mask != "all": harm = mask_harmonics(harm, h_mask, h_mask_k, n_harmonics=h_n)

    waveform_fn = make_waveform_fn("wavefold", wavefold) if wavefold > 0 else None

    h_audio = harmonic_oscillator_bent(
        f0_hz, harm, controls["global_amp"],
        waveform_fn=waveform_fn,
        fm_depth=fm_depth, fm_ratio=fm_ratio,
        inharmonicity=h_inharmonicity, h_n=h_n,
        sample_rate=model.sample_rate, hop_length=model.hop_length,
    )
    n_audio = model.filtered_noise(controls["noise_mag"])
    dry = harmonic_gain * h_audio + noise_gain * n_audio

    ir = get_ir(model)
    if ir_reverse:
        ir = ir.flip(0)
    audio = apply_ir(dry, ir)

    return dict(
        audio=audio, dry=dry,
        harmonic_audio=h_audio, noise_audio=n_audio,
        harmonic_dist=harm, noise_mag=controls["noise_mag"],
        global_amp=controls["global_amp"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def specshow(audio: torch.Tensor, sr: int = 16000,
             title: str = "", ax=None, fmax: float = 8000.0) -> None:
    """Log-frequency spectrogram using librosa."""
    import librosa
    import librosa.display
    wav = audio.squeeze().detach().numpy()
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=256,
                              x_axis="time", y_axis="log",
                              ax=ax, fmax=fmax)
    ax.set_title(title)
    if own:
        plt.tight_layout()
        plt.show()


def play(audio: torch.Tensor, sr: int = 16000,
         normalize: bool = True) -> None:
    """Display an inline IPython Audio widget."""
    wav = audio.squeeze().detach().numpy()
    if normalize and np.abs(wav).max() > 1e-8:
        wav = wav / np.abs(wav).max() * 0.9
    display(Audio(wav, rate=sr))


def heatmap(harmonic_dist: torch.Tensor, title: str = "") -> None:
    """Per-frame harmonic amplitude distribution as a 2-D heatmap."""
    data = harmonic_dist.squeeze().detach().numpy()   # (N, K)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(data.T, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(im, ax=ax)
    ax.set(xlabel="Frame", ylabel="Harmonic #", title=title)
    plt.tight_layout()
    plt.show()
