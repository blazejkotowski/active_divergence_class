"""
Inspiring-set synthesis for animal vocalizations via a deliberately limited synth.

Based on Hagiwara et al. (2022): an intentionally constrained synthesizer is
optimised per target sample.  Because it cannot faithfully reproduce the target,
optimization produces a creative approximation — the inspiring-set effect.

The synth used here is a two-operator FM synthesizer with additive amplitude
modulation (AM/tremolo), covering:
  - 16 pitch control points (coarse trajectory)
  - FM: modulation depth + modulator frequency
  - AM: tremolo depth + tremolo rate  (gives rhythmic character — chirp/croak patterns)
  - 16 amplitude envelope control points
  - White noise floor

All raw parameters live in unconstrained ℝ and are mapped to musical ranges via
sigmoid/tanh/exp — NO hard clamps that kill gradients.  This is the critical fix
vs. the naive MIDI-clamping approach.

Deliberate limitations that create divergence:
  - 16 pitch points cannot capture rapid bird trills (too coarse)
  - Max pitch ~1047 Hz (MIDI 84) — cannot reach 2–4 kHz bird calls directly;
    the synth echoes them an octave below (which IS the inspiring-set gap)
  - Single FM operator cannot reproduce multi-formant frog calls
  - White noise floor is not shaped like breath or wing noise
"""

import os
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
ANIMAL_CATEGORIES = {"cat", "chirping_birds", "frog", "crickets", "crow", "insects", "dog"}


# ─────────────────────────────────────────────────────────────────────────────
# Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

class AnimalSynth(nn.Module):
    """
    Two-operator FM + AM synthesizer for animal vocalization approximation.

    Signal chain::

        pitch trajectory → FM carrier → × AM tremolo → × amplitude envelope → + noise → audio

    All parameters use differentiable mappings (tanh/sigmoid/exp) so gradients
    flow everywhere — no hard clamps.

    Parameter mappings (raw ∈ ℝ → musical range):

        pitch_raw    (n_ctrl,)  tanh → MIDI [40, 84] → Hz  (E2 = 82 Hz to C6 = 1047 Hz)
        fm_depth_raw (1,)       sigmoid × 8 → FM index [0, 8]
        fm_rate_raw  (1,)       5 × exp(clamp(·, −3, 3)) → rate [0.25, 100] Hz
        am_depth_raw (1,)       sigmoid → tremolo depth [0, 1]
        am_rate_raw  (1,)       5 × exp(clamp(·, −3, 3)) → rate [0.25, 100] Hz
        amp_raw      (n_ctrl,)  sigmoid → amplitude envelope [0, 1]
        noise_raw    (1,)       sigmoid × 0.4 → noise gain [0, 0.4]

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

        # Pitch trajectory: raw → tanh → MIDI [40, 84] → Hz
        # Initialize with small random variation so optimizer has starting spread
        self.pitch_raw   = nn.Parameter(torch.randn(n_ctrl) * 0.5)  # → ~62 MIDI ± variation

        # FM
        self.fm_depth_raw = nn.Parameter(torch.tensor(-1.0))  # → sigmoid(-1)*8 ≈ 2.1 index
        self.fm_rate_raw  = nn.Parameter(torch.tensor(0.0))   # → 5*exp(0) = 5 Hz modulator

        # AM / tremolo
        self.am_depth_raw = nn.Parameter(torch.tensor(-2.0))  # → sigmoid(-2) ≈ 0.12 (mild AM)
        self.am_rate_raw  = nn.Parameter(torch.tensor(1.0))   # → 5*exp(1) ≈ 13.6 Hz chirp rate

        # Amplitude envelope
        # Initialize with a gentle onset/decay shape using a ramp
        init_amp = torch.linspace(-2.0, -0.5, n_ctrl) + torch.randn(n_ctrl) * 0.3
        self.amp_raw = nn.Parameter(init_amp)

        # Noise floor
        self.noise_raw = nn.Parameter(torch.tensor(-3.0))  # → sigmoid(-3)*0.4 ≈ 0.02 (quiet)

        # Fixed noise vector (seed-stable for gradient flow through noise_raw)
        self.register_buffer("_noise", torch.randn(self.T))

    # ── Constrained parameter accessors ──────────────────────────────────────
    def pitch_hz(self) -> torch.Tensor:
        """Returns (n_ctrl,) pitch trajectory in Hz, fully differentiable."""
        midi = 62.0 + 22.0 * torch.tanh(self.pitch_raw / 5.0)  # MIDI [40, 84]
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def fm_depth(self) -> torch.Tensor:
        return 8.0 * torch.sigmoid(self.fm_depth_raw)           # [0, 8]

    def fm_rate_hz(self) -> torch.Tensor:
        return 5.0 * torch.exp(self.fm_rate_raw.clamp(-3.0, 3.0))  # [0.25, 100] Hz

    def am_depth(self) -> torch.Tensor:
        return torch.sigmoid(self.am_depth_raw)                  # [0, 1]

    def am_rate_hz(self) -> torch.Tensor:
        return 5.0 * torch.exp(self.am_rate_raw.clamp(-3.0, 3.0))  # [0.25, 100] Hz

    def amplitude_envelope(self) -> torch.Tensor:
        """(n_ctrl,) amplitude envelope control points in [0, 1]."""
        return torch.sigmoid(self.amp_raw)

    def noise_gain(self) -> torch.Tensor:
        return 0.4 * torch.sigmoid(self.noise_raw)               # [0, 0.4]

    def forward(self) -> torch.Tensor:
        T, sr = self.T, self.sr
        device = self.pitch_raw.device
        t = torch.linspace(0.0, self.duration, T, device=device)

        # ── Pitch trajectory ──────────────────────────────────────────────────
        f0_ctrl = self.pitch_hz()  # (n_ctrl,)
        f0_hz = F.interpolate(
            f0_ctrl.unsqueeze(0).unsqueeze(0), T, mode="linear", align_corners=True
        ).squeeze()  # (T,)

        # ── FM synthesis ──────────────────────────────────────────────────────
        fm_d = self.fm_depth()
        fm_r = self.fm_rate_hz()
        # Phase of FM carrier: integrate f0 + FM phase modulation added directly
        carrier_phase = 2.0 * torch.pi * torch.cumsum(f0_hz / sr, dim=0)
        fm_phase      = fm_d * torch.sin(2.0 * torch.pi * fm_r * t)
        carrier       = torch.sin(carrier_phase + fm_phase)

        # ── Amplitude modulation (tremolo / chirp pattern) ────────────────────
        am_d   = self.am_depth()
        am_r   = self.am_rate_hz()
        # AM: (1 + α·sin(2π·am_rate·t)) — adds rhythmic envelope variation
        am_env = 1.0 + am_d * torch.sin(2.0 * torch.pi * am_r * t)

        # ── Amplitude envelope (coarse shape) ─────────────────────────────────
        amp_ctrl = self.amplitude_envelope()  # (n_ctrl,) in [0, 1]
        envelope = F.interpolate(
            amp_ctrl.unsqueeze(0).unsqueeze(0), T, mode="linear", align_corners=True
        ).squeeze()  # (T,)

        # ── Noise floor ───────────────────────────────────────────────────────
        noise = self._noise.to(device) * self.noise_gain()

        # ── Mix and normalise ─────────────────────────────────────────────────
        audio = envelope * am_env * carrier + noise
        peak  = audio.abs().max().clamp(min=1e-8)
        return audio / peak * 0.9

    def get_param_dict(self) -> dict:
        """Human-readable parameter summary."""
        with torch.no_grad():
            f0s = self.pitch_hz().cpu().tolist()
            return {
                "pitch_hz_mean": float(sum(f0s) / len(f0s)),
                "pitch_hz_range": (min(f0s), max(f0s)),
                "fm_depth": self.fm_depth().item(),
                "fm_rate_hz": self.fm_rate_hz().item(),
                "am_depth": self.am_depth().item(),
                "am_rate_hz": self.am_rate_hz().item(),
                "noise_gain": self.noise_gain().item(),
            }

    def perturb(
        self,
        pitch_semitones: float = 0.0,
        fm_depth_add: float = 0.0,
        fm_rate_mult: float = 1.0,
        am_depth_add: float = 0.0,
        am_rate_mult: float = 1.0,
        noise_add: float = 0.0,
    ) -> "AnimalSynth":
        """
        Return a new synth with parameters perturbed for interactive exploration.
        Operates on the mapped (musical) values so sliders feel linear.
        """
        clone = AnimalSynth(self.n_ctrl, self.sr, self.duration)
        clone.load_state_dict(self.state_dict())
        with torch.no_grad():
            # Shift all pitch control points by semitone offset (in raw space)
            # pitch_hz = 440 * 2^((62 + 22*tanh(raw/5) - 69)/12)
            # Adding N semitones to MIDI shifts raw: Δraw ≈ Δmidi/22 * 5/tanh'
            # Approximate: shift raw proportionally
            clone.pitch_raw.add_(pitch_semitones * 5.0 / 22.0)
            # FM depth: shift in sigmoid space
            current_fd = self.fm_depth().item()
            new_fd = (current_fd + fm_depth_add).clip(0, 8)
            # Invert sigmoid: raw = logit(new_fd/8)
            eps = 1e-6
            clone.fm_depth_raw.data.fill_(
                torch.logit(torch.tensor(new_fd / 8.0).clamp(eps, 1 - eps)).item()
            )
            # FM rate: multiply in log space
            current_fr = self.fm_rate_hz().item()
            new_fr = (current_fr * fm_rate_mult).clip(0.25, 100)
            clone.fm_rate_raw.data.fill_(float(torch.tensor(new_fr / 5.0).log()))
            # AM depth: additive
            current_ad = self.am_depth().item()
            new_ad = (current_ad + am_depth_add).clip(0, 1)
            clone.am_depth_raw.data.fill_(torch.logit(torch.tensor(new_ad).clamp(eps, 1 - eps)).item())
            # AM rate: multiply
            current_ar = self.am_rate_hz().item()
            new_ar = (current_ar * am_rate_mult).clip(0.25, 100)
            clone.am_rate_raw.data.fill_(float(torch.tensor(new_ar / 5.0).log()))
            # Noise
            clone.noise_raw.add_(noise_add)
        return clone


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                   fft_sizes: tuple = (1024, 512, 256, 128)) -> torch.Tensor:
    """Multi-scale spectral distance, linear + log magnitude."""
    loss = pred.new_zeros(1)
    for n in fft_sizes:
        hop = max(n // 4, 1)
        win = torch.hann_window(n, device=pred.device)
        S = lambda x: torch.stft(
            x, n_fft=n, hop_length=hop, win_length=n, window=win, return_complex=True
        ).abs()
        Sp, St = S(pred), S(target)
        loss = loss + (Sp - St).abs().mean()
        loss = loss + (Sp.clamp(1e-7).log() - St.clamp(1e-7).log()).abs().mean()
    return loss


def fit_synth_to_target(
    target_audio: torch.Tensor,
    n_iter: int = 400,
    lr: float = 0.08,
    n_ctrl: int = 16,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> tuple["AnimalSynth", list[float]]:
    """
    Optimise AnimalSynth parameters to match ``target_audio`` via spectral loss.

    This is the Hagiwara-style per-sample fit.  The synthesizer is intentionally
    limited so it cannot copy the target exactly — it produces a creative approximation.

    The optimizer uses cosine-annealed Adam for stable convergence.  Parameters
    are unconstrained (ℝ) with differentiable musical-range mappings, so gradients
    flow everywhere and the optimizer can explore the full parameter space.

    Args:
        target_audio: (T,) mono audio tensor at ``sample_rate``
        n_iter:       number of Adam iterations (400 recommended)
        lr:           initial learning rate (0.08 recommended)
        n_ctrl:       number of pitch/amp control points
        sample_rate:  audio sample rate (Hz)
        verbose:      print loss every 50 steps

    Returns:
        (synth, loss_history)
    """
    duration = len(target_audio) / sample_rate
    synth = AnimalSynth(n_ctrl=n_ctrl, sample_rate=sample_rate, duration=duration)

    optimizer = torch.optim.Adam(synth.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr * 0.05)
    losses = []

    target_audio = target_audio.detach()
    T = len(target_audio)

    for step in range(n_iter):
        optimizer.zero_grad()
        pred = synth()
        loss = _spectral_loss(pred[:T], target_audio[:T])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(synth.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if verbose and (step % 80 == 0 or step == n_iter - 1):
            p = synth.get_param_dict()
            print(f"  step {step:3d}/{n_iter}  loss={loss.item():.3f}  "
                  f"f0={p['pitch_hz_mean']:.0f}Hz  "
                  f"fm={p['fm_depth']:.1f}@{p['fm_rate_hz']:.1f}Hz  "
                  f"am={p['am_depth']:.2f}@{p['am_rate_hz']:.1f}Hz")

    return synth, losses


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (ESC-50)
# ─────────────────────────────────────────────────────────────────────────────

def load_esc50_animals(
    esc50_dir: str,
    categories: set | None = None,
    max_per_category: int = 1,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> list[dict]:
    """
    Load a small set of animal vocalization clips from ESC-50.

    Returns a list of dicts: {'audio': (T,) tensor, 'category': str, 'filename': str}

    Args:
        esc50_dir:        path to ESC-50 root (contains audio/, meta/esc50.csv)
        categories:       which categories to include. Default: cat, frog, chirping_birds, crickets
        max_per_category: cap samples per category (1 keeps the demo concise)
        duration:         clip length to return (seconds), trimmed from start
        sample_rate:      target sample rate
    """
    if categories is None:
        categories = {"cat", "frog", "chirping_birds", "crickets"}

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
            waveform = waveform.squeeze(0)  # (T_full,)

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
