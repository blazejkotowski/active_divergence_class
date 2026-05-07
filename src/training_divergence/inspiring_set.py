"""
Inspiring-set synthesis for animal vocalizations via a deliberately limited synth.

Based on the core idea of Hagiwara et al. (2022): an intentionally constrained
synthesizer is optimized per target sample.  Because it cannot faithfully reproduce
the target, optimization produces a creative approximation — the inspiring-set effect.

The synth used here is a single-operator FM voice with:
  - 16 pitch control points (coarse trajectory, linearly interpolated)
  - 16 amplitude control points (coarse envelope)
  - FM modulation (depth + modulator frequency)
  - White noise floor

These constraints prevent exact reproduction of, e.g., rapid bird trills or frog calls,
so the result is a synthesized character that echoes the original without copying it.
"""

import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
ANIMAL_CATEGORIES = {"chirping_birds", "crow", "frog", "insects", "crickets", "dog"}


# ─────────────────────────────────────────────────────────────────────────────
# Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

class AnimalSynth(nn.Module):
    """
    Intentionally limited FM+noise synthesizer.

    All parameters are nn.Parameters optimized via gradient descent.
    The synthesizer is deliberately simple:
      - No rapid pitch modulation beyond the 16-point trajectory
      - No complex filter or resonator
      - No temporal structure beyond the amplitude envelope
    This limitation is the inspiring-set mechanism: optimization produces
    a plausible but stylised version of the target.

    Args:
        n_ctrl:      number of pitch and amplitude control points
        sample_rate: audio sample rate (Hz)
        duration:    clip duration (seconds)
    """

    def __init__(self, n_ctrl: int = 16, sample_rate: int = 16000, duration: float = 2.0):
        super().__init__()
        self.sr = sample_rate
        self.duration = duration
        self.T = int(duration * sample_rate)
        self.n_ctrl = n_ctrl

        # Pitch trajectory: MIDI note numbers (60 = C4 = 261.6 Hz)
        self.pitch_ctrl = nn.Parameter(torch.full((n_ctrl,), 60.0))

        # Amplitude envelope control points (log scale, softplus activiation in forward)
        self.amp_ctrl = nn.Parameter(torch.zeros(n_ctrl))

        # FM parameters
        self.fm_depth = nn.Parameter(torch.tensor(1.0))       # modulation index β
        self.fm_rate_hz = nn.Parameter(torch.tensor(5.0))     # modulator freq in Hz

        # Noise floor
        self.noise_level = nn.Parameter(torch.tensor(-2.0))   # log-scale noise gain

        # Fixed noise vector (seed fixed so gradients flow only through noise_level)
        self.register_buffer("_noise", torch.randn(self.T))

    def forward(self) -> torch.Tensor:
        T, sr = self.T, self.sr
        t = torch.linspace(0, self.duration, T, device=self.pitch_ctrl.device)

        # ── Pitch trajectory ───────────────────────────────────────────────
        # Interpolate n_ctrl MIDI values → T-length pitch trajectory
        pitch_up = F.interpolate(
            self.pitch_ctrl.unsqueeze(0).unsqueeze(0), T, mode="linear", align_corners=True
        ).squeeze()  # (T,)
        f0_hz = 440.0 * (2.0 ** ((pitch_up.clamp(20, 100) - 69.0) / 12.0))

        # ── FM modulation ──────────────────────────────────────────────────
        fm_depth = torch.abs(self.fm_depth).clamp(max=12.0)
        fm_rate = torch.abs(self.fm_rate_hz).clamp(max=200.0)

        mod_phase = 2.0 * torch.pi * fm_rate * t
        # Instantaneous carrier freq with FM
        inst_freq = f0_hz * (1.0 + fm_depth * torch.sin(mod_phase) / (2.0 * torch.pi))
        carrier_phase = 2.0 * torch.pi * torch.cumsum(inst_freq / sr, dim=0)
        carrier = torch.sin(carrier_phase)

        # ── Amplitude envelope ─────────────────────────────────────────────
        amp_up = F.interpolate(
            self.amp_ctrl.unsqueeze(0).unsqueeze(0), T, mode="linear", align_corners=True
        ).squeeze()  # (T,)
        envelope = torch.sigmoid(amp_up)  # smooth, bounded [0,1]

        # ── Noise floor ────────────────────────────────────────────────────
        noise_gain = torch.sigmoid(self.noise_level) * 0.5  # 0..0.5
        noise = self._noise.to(carrier.device) * noise_gain

        # ── Mix and normalise ──────────────────────────────────────────────
        audio = envelope * carrier + noise
        peak = audio.abs().max().clamp(min=1e-8)
        return audio / peak * 0.9

    def get_param_dict(self) -> dict:
        """Return current parameters as a human-readable dict."""
        with torch.no_grad():
            return {
                "pitch_ctrl_midi": self.pitch_ctrl.detach().cpu().tolist(),
                "fm_depth": torch.abs(self.fm_depth).item(),
                "fm_rate_hz": torch.abs(self.fm_rate_hz).item(),
                "noise_level_gain": (torch.sigmoid(self.noise_level) * 0.5).item(),
            }

    def perturb(self, pitch_shift: float = 0.0, fm_depth_add: float = 0.0,
                fm_rate_mult: float = 1.0, noise_gain_add: float = 0.0) -> "AnimalSynth":
        """Return a new synth with parameters perturbed — for interactive exploration."""
        clone = AnimalSynth(self.n_ctrl, self.sr, self.duration)
        clone.load_state_dict(self.state_dict())
        with torch.no_grad():
            clone.pitch_ctrl += pitch_shift
            clone.fm_depth += fm_depth_add
            clone.fm_rate_hz *= fm_rate_mult
            clone.noise_level += noise_gain_add
        return clone


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _multi_scale_spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                                fft_sizes=(1024, 512, 256, 128)) -> torch.Tensor:
    """Lightweight multi-scale spectral loss for per-sample fitting."""
    loss = torch.zeros(1, device=pred.device)
    for n in fft_sizes:
        hop = max(n // 4, 1)
        win = torch.hann_window(n, device=pred.device)
        S = lambda x: torch.stft(x, n_fft=n, hop_length=hop, win_length=n,
                                  window=win, return_complex=True).abs()
        Sp, St = S(pred), S(target)
        loss = loss + (Sp - St).abs().mean()
        loss = loss + (Sp.clamp(1e-7).log() - St.clamp(1e-7).log()).abs().mean()
    return loss


def fit_synth_to_target(
    target_audio: torch.Tensor,
    n_iter: int = 300,
    lr: float = 5e-2,
    n_ctrl: int = 16,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> tuple["AnimalSynth", list[float]]:
    """
    Optimise AnimalSynth parameters to match ``target_audio`` via spectral loss.

    This is the Hagiwara-style per-sample fit.  The synthesizer is intentionally
    limited so it cannot copy the target exactly — it produces a creative approximation.

    Args:
        target_audio: (T,) mono audio tensor at ``sample_rate``
        n_iter:       number of Adam iterations
        lr:           learning rate
        n_ctrl:       number of pitch/amp control points in the synth
        sample_rate:  audio sample rate (Hz)
        verbose:      print loss every 50 steps

    Returns:
        (synth, loss_history)  where synth.forward() gives the best approximation.
    """
    duration = len(target_audio) / sample_rate
    synth = AnimalSynth(n_ctrl=n_ctrl, sample_rate=sample_rate, duration=duration)

    # Initialise pitch near the detected centre of mass of the target spectrum
    with torch.no_grad():
        spec = torch.fft.rfft(target_audio).abs()
        freqs = torch.fft.rfftfreq(len(target_audio)) * sample_rate
        centroid = (freqs * spec).sum() / spec.sum().clamp(min=1e-8)
        centroid = centroid.clamp(80, 4000)
        midi_init = 12.0 * torch.log2(centroid / 440.0) + 69.0
        synth.pitch_ctrl.fill_(midi_init.item())

    optimizer = torch.optim.Adam(synth.parameters(), lr=lr)
    losses = []

    target_audio = target_audio.detach()

    for step in range(n_iter):
        optimizer.zero_grad()
        pred = synth()
        # Trim or pad to match target length
        T = min(pred.shape[0], target_audio.shape[0])
        loss = _multi_scale_spectral_loss(pred[:T], target_audio[:T])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(synth.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if verbose and (step % 50 == 0 or step == n_iter - 1):
            print(f"  step {step:3d}/{n_iter} | loss {loss.item():.4f}")

    return synth, losses


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (ESC-50)
# ─────────────────────────────────────────────────────────────────────────────

def load_esc50_animals(
    esc50_dir: str,
    categories: set[str] | None = None,
    max_per_category: int = 2,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> list[dict]:
    """
    Load a small set of animal vocalization clips from ESC-50.

    Returns a list of dicts with keys:
        'audio':    (T,) mono float32 tensor at ``sample_rate``
        'category': str (e.g. 'chirping_birds')
        'filename': str

    Args:
        esc50_dir:        path to ESC-50 root (contains audio/, meta/esc50.csv)
        categories:       which categories to include (default: ANIMAL_CATEGORIES)
        max_per_category: cap samples per category
        duration:         clip length to return (seconds)
        sample_rate:      target sample rate
    """
    if categories is None:
        categories = ANIMAL_CATEGORIES

    meta_path = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    selected: list[dict] = []
    counts: dict[str, int] = {}

    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            if cat not in categories:
                continue
            if counts.get(cat, 0) >= max_per_category:
                continue
            path = os.path.join(audio_dir, row["filename"])
            if not os.path.exists(path):
                continue

            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            waveform = waveform.squeeze(0)  # (T,)

            # Trim to ``duration``
            n_target = int(duration * sample_rate)
            if waveform.shape[0] >= n_target:
                waveform = waveform[:n_target]
            else:
                waveform = F.pad(waveform, (0, n_target - waveform.shape[0]))

            # Peak-normalise
            peak = waveform.abs().max().clamp(min=1e-8)
            waveform = waveform / peak * 0.9

            selected.append({"audio": waveform, "category": cat, "filename": row["filename"]})
            counts[cat] = counts.get(cat, 0) + 1

    return selected
