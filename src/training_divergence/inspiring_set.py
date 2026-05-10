"""
Inspiring-set synthesis for animal vocalizations via a limited modular synthesizer.

Based on Hagiwara et al. (2022) "Modeling Animal Vocalizations through Synthesizers":
optimise a deliberately constrained modular synthesizer per target clip so it cannot
faithfully reproduce the target — the gap IS the creative approximation.

Synthesizer: VoiceSynth — a pure-PyTorch re-implementation of the TorchSynth Voice
architecture used in the paper (bypasses TorchSynth's broken pytorch-lightning
dependency on Python 3.12).

Architecture (matches TorchSynth Voice):
  keyboard:     MIDI pitch + note duration
  adsr_1/2:     main envelopes (pitch / amp shaping)
  lfo_1/2:      LFOs, each with rate_adsr + amp_adsr
  mod_matrix:   4 signals × 5 destinations (VCO1 pitch/amp, VCO2 pitch/amp, noise amp)
  vco_1:        SineVCO (sinusoidal oscillator)
  vco_2:        SawVCO (sawtooth via harmonic series)
  noise:        white noise with amp envelope
  mixer:        weighted sum of VCO1, VCO2, Noise

Optimization: Adam, 200 iterations, lr=0.001 (exactly as in the paper).
Loss: multi-scale STFT (linear + log magnitude).
"""

import os
import csv
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
DURATION    = 2.0
ANIMAL_CATEGORIES = {"cat", "chirping_birds", "frog", "crickets", "crow", "insects", "dog"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adsr_envelope(
    attack: torch.Tensor,   # seconds [0, 2]
    decay: torch.Tensor,    # seconds [0, 2]
    sustain: torch.Tensor,  # level   [0, 1]
    release: torch.Tensor,  # seconds [0, 4]
    note_on: torch.Tensor,  # seconds [0, duration]
    T: int,
    sr: int,
) -> torch.Tensor:
    """Differentiable piecewise-linear ADSR envelope, shape (T,)."""
    total = T / sr
    device = attack.device

    # All keypoints as 0-dim scalars
    a  = attack.squeeze().clamp(0.0, total)
    d  = decay.squeeze().clamp(0.0, total)
    s  = sustain.squeeze().clamp(0.0, 1.0)
    r  = release.squeeze().clamp(0.0, total)
    no = note_on.squeeze().clamp(0.0, total)

    t1 = a
    t2 = (a + d).clamp(0.0, total)
    t3 = torch.max(no, t2)
    t4 = (t3 + r).clamp(0.0, total)

    t = torch.linspace(0.0, total, T, device=device)

    def seg(ta, tb, va, vb):
        dt = (tb - ta).clamp(min=1e-6)
        alpha = ((t - ta) / dt).clamp(0.0, 1.0)
        mask = ((t >= ta) & (t < tb)).float()
        return mask * (va + alpha * (vb - va))

    zero = t.new_zeros(())
    one  = t.new_ones(())

    env = (
        seg(zero, t1, zero, one)   # attack
        + seg(t1, t2, one, s)      # decay
        + seg(t2, t3, s, s)        # sustain
        + seg(t3, t4, s, zero)     # release
    )
    # Clamp to [0,1] — rounding errors can push slightly outside
    return env.clamp(0.0, 1.0)


def _saw_wave(phase: torch.Tensor, n_harmonics: int = 8) -> torch.Tensor:
    """Differentiable sawtooth via truncated Fourier series."""
    out = torch.zeros_like(phase)
    for k in range(1, n_harmonics + 1):
        out = out + ((-1.0) ** (k + 1)) * torch.sin(k * phase) / k
    return out * (2.0 / math.pi)


def _lfo(rate_hz: torch.Tensor, T: int, sr: int) -> torch.Tensor:
    """Sinusoidal LFO at given rate, output in [-1, 1]."""
    t = torch.linspace(0.0, T / sr, T, device=rate_hz.device)
    return torch.sin(2.0 * math.pi * rate_hz * t)


# ─────────────────────────────────────────────────────────────────────────────
# VoiceSynth  (TorchSynth Voice re-implementation)
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSynth(nn.Module):
    """
    Pure-PyTorch Voice synthesizer matching Hagiwara et al. (2022).

    Modules (≈ 74 parameters, matching TorchSynth Voice):
        keyboard      — MIDI pitch + note-on duration          (2 params)
        adsr_1/2      — main envelopes                         (5 × 2 = 10)
        lfo_1/2       — LFOs                                   (2 × 2 = 4)
        lfo_1/2_amp   — amplitude ADSRs for LFOs               (5 × 2 = 10)
        lfo_1/2_rate  — rate ADSRs for LFOs                    (5 × 2 = 10)
        mod_matrix    — 4 inputs × 5 destinations              (20)
        vco_1 (sine)  — sinusoidal VCO                         (3)
        vco_2 (saw)   — sawtooth VCO                           (4)
        noise         — white noise generator                  (2)
        mixer         — audio output mixer                     (3)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        super().__init__()
        self.sr = sample_rate
        self.duration = duration
        self.T = int(duration * sample_rate)

        # ── Keyboard ─────────────────────────────────────────────────────────
        # midi_f0: raw → sigmoid × 127  (MIDI note 0–127 → Hz)
        self.kbd_midi_raw     = nn.Parameter(torch.zeros(1))   # → MIDI ~64 (≈330 Hz)
        # note_on_duration: raw → sigmoid × duration
        self.kbd_duration_raw = nn.Parameter(torch.zeros(1))   # → 50% note-on

        # ── ADSR 1 (pitch/amp envelope 1) ────────────────────────────────────
        self.adsr1_a_raw = nn.Parameter(torch.tensor(-2.0))    # → ~0.07s attack
        self.adsr1_d_raw = nn.Parameter(torch.tensor(0.0))     # → ~1.0s decay
        self.adsr1_s_raw = nn.Parameter(torch.tensor(0.0))     # → 0.5 sustain
        self.adsr1_r_raw = nn.Parameter(torch.tensor(0.0))     # → ~2.0s release
        self.adsr1_alpha = nn.Parameter(torch.tensor(0.0))     # curve shape

        # ── ADSR 2 ───────────────────────────────────────────────────────────
        self.adsr2_a_raw = nn.Parameter(torch.tensor(0.0))
        self.adsr2_d_raw = nn.Parameter(torch.tensor(0.0))
        self.adsr2_s_raw = nn.Parameter(torch.tensor(0.0))
        self.adsr2_r_raw = nn.Parameter(torch.tensor(0.0))
        self.adsr2_alpha = nn.Parameter(torch.tensor(0.0))

        # ── LFO 1 ────────────────────────────────────────────────────────────
        self.lfo1_rate_raw  = nn.Parameter(torch.tensor(0.0))  # → ~1 Hz
        self.lfo1_shape_raw = nn.Parameter(torch.tensor(0.0))  # blend

        # LFO1 amp ADSR
        self.lfo1_amp_a_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_amp_d_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_amp_s_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_amp_r_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_amp_alpha = nn.Parameter(torch.tensor(0.0))

        # LFO1 rate ADSR
        self.lfo1_rate_a_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_rate_d_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_rate_s_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_rate_r_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo1_rate_alpha = nn.Parameter(torch.tensor(0.0))

        # ── LFO 2 ────────────────────────────────────────────────────────────
        self.lfo2_rate_raw  = nn.Parameter(torch.tensor(1.0))  # → ~2.7 Hz
        self.lfo2_shape_raw = nn.Parameter(torch.tensor(0.0))

        # LFO2 amp ADSR
        self.lfo2_amp_a_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_amp_d_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_amp_s_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_amp_r_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_amp_alpha = nn.Parameter(torch.tensor(0.0))

        # LFO2 rate ADSR
        self.lfo2_rate_a_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_rate_d_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_rate_s_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_rate_r_raw = nn.Parameter(torch.tensor(0.0))
        self.lfo2_rate_alpha = nn.Parameter(torch.tensor(0.0))

        # ── Modulation Matrix (4 inputs → 5 destinations) ─────────────────────
        # Inputs:  adsr_1, adsr_2, lfo_1, lfo_2
        # Outputs: vco1_pitch, vco1_amp, vco2_pitch, vco2_amp, noise_amp
        self.mod_matrix_raw = nn.Parameter(torch.randn(4, 5) * 0.1)  # (4, 5)

        # ── VCO 1 (Sine) ─────────────────────────────────────────────────────
        self.vco1_detune_raw = nn.Parameter(torch.tensor(0.0))   # → ±12 semitones
        self.vco1_gain_raw   = nn.Parameter(torch.tensor(0.0))   # → sigmoid
        self.vco1_phase_raw  = nn.Parameter(torch.tensor(0.0))   # → 2π phase offset

        # ── VCO 2 (Saw) ──────────────────────────────────────────────────────
        self.vco2_detune_raw = nn.Parameter(torch.tensor(0.5))   # init slightly detuned
        self.vco2_gain_raw   = nn.Parameter(torch.tensor(0.0))
        self.vco2_shape_raw  = nn.Parameter(torch.tensor(0.0))   # 0=sine, 1=saw
        self.vco2_phase_raw  = nn.Parameter(torch.tensor(0.0))

        # ── Noise ─────────────────────────────────────────────────────────────
        self.noise_gain_raw  = nn.Parameter(torch.tensor(-1.0))  # → ~0.27
        self.noise_tilt_raw  = nn.Parameter(torch.tensor(0.0))   # spectral colour

        # ── Audio Mixer ───────────────────────────────────────────────────────
        # Final blend of VCO1, VCO2, Noise (softmax)
        self.mixer_raw = nn.Parameter(torch.tensor([0.5, 0.0, -1.0]))

        # Fixed noise buffer
        self.register_buffer("_noise_buf", torch.randn(self.T))

    # ── Parameter accessors ──────────────────────────────────────────────────

    def _adsr_params(self, a_raw, d_raw, s_raw, r_raw):
        a = 2.0  * torch.sigmoid(a_raw)         # [0, 2] s
        d = 2.0  * torch.sigmoid(d_raw)         # [0, 2] s
        s = torch.sigmoid(s_raw)                 # [0, 1]
        r = 4.0  * torch.sigmoid(r_raw)         # [0, 4] s
        return a, d, s, r

    def _kbd_midi_hz(self):
        midi = 127.0 * torch.sigmoid(self.kbd_midi_raw)   # [0, 127]
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def _kbd_note_on(self):
        return self.duration * torch.sigmoid(self.kbd_duration_raw)

    def _lfo_rate(self, rate_raw, rate_adsr_env):
        """LFO instantaneous rate in Hz, modulated by its rate ADSR."""
        base = 20.0 * torch.sigmoid(rate_raw)             # [0, 20] Hz base
        mod  = 1.0 + rate_adsr_env                        # 1× to 2× range
        return (base * mod).clamp(0.1, 40.0)

    def _mod_matrix(self):
        return torch.sigmoid(self.mod_matrix_raw)          # (4, 5) in [0, 1]

    def forward(self) -> torch.Tensor:
        T, sr = self.T, self.sr
        device = self.kbd_midi_raw.device
        note_on = self._kbd_note_on()
        f0_hz   = self._kbd_midi_hz()

        # ── ADSRs ──────────────────────────────────────────────────────────
        def adsr(a_r, d_r, s_r, r_r):
            a, d, s, r = self._adsr_params(a_r, d_r, s_r, r_r)
            return _adsr_envelope(a, d, s, r, note_on, T, sr)

        adsr1 = adsr(self.adsr1_a_raw, self.adsr1_d_raw, self.adsr1_s_raw, self.adsr1_r_raw)
        adsr2 = adsr(self.adsr2_a_raw, self.adsr2_d_raw, self.adsr2_s_raw, self.adsr2_r_raw)

        lfo1_amp_env  = adsr(self.lfo1_amp_a_raw, self.lfo1_amp_d_raw,
                             self.lfo1_amp_s_raw,  self.lfo1_amp_r_raw)
        lfo2_amp_env  = adsr(self.lfo2_amp_a_raw, self.lfo2_amp_d_raw,
                             self.lfo2_amp_s_raw,  self.lfo2_amp_r_raw)
        lfo1_rate_env = adsr(self.lfo1_rate_a_raw, self.lfo1_rate_d_raw,
                             self.lfo1_rate_s_raw,  self.lfo1_rate_r_raw)
        lfo2_rate_env = adsr(self.lfo2_rate_a_raw, self.lfo2_rate_d_raw,
                             self.lfo2_rate_s_raw,  self.lfo2_rate_r_raw)

        # ── LFOs (rate-modulated) ──────────────────────────────────────────
        lfo1_rate = self._lfo_rate(self.lfo1_rate_raw, lfo1_rate_env)
        lfo2_rate = self._lfo_rate(self.lfo2_rate_raw, lfo2_rate_env)

        t = torch.linspace(0.0, self.duration, T, device=device)
        lfo1_sig = torch.sin(2.0 * math.pi * lfo1_rate * t) * lfo1_amp_env
        lfo2_sig = torch.sin(2.0 * math.pi * lfo2_rate * t) * lfo2_amp_env

        # ── Modulation matrix ─────────────────────────────────────────────
        # (4, 5): each row is one signal, columns are destinations
        M = self._mod_matrix()
        sigs = torch.stack([adsr1, adsr2, lfo1_sig, lfo2_sig], dim=0)  # (4, T)
        # mod_dest: (5, T)
        mod_dest = torch.einsum("si,st->it", M, sigs)
        vco1_pitch_mod = mod_dest[0]   # (T,)
        vco1_amp_mod   = mod_dest[1]
        vco2_pitch_mod = mod_dest[2]
        vco2_amp_mod   = mod_dest[3]
        noise_amp_mod  = mod_dest[4]

        # ── VCO 1 (Sine) ──────────────────────────────────────────────────
        vco1_detune = 12.0 * torch.tanh(self.vco1_detune_raw)  # ±12 semitones
        vco1_f = f0_hz * (2.0 ** (vco1_detune / 12.0)) * (1.0 + 0.1 * vco1_pitch_mod)
        vco1_phase = 2.0 * math.pi * torch.cumsum(vco1_f / sr, dim=0)
        vco1_init  = 2.0 * math.pi * torch.sigmoid(self.vco1_phase_raw)
        vco1_raw   = torch.sin(vco1_phase + vco1_init)
        vco1_gain  = torch.sigmoid(self.vco1_gain_raw)
        vco1_out   = vco1_raw * (vco1_gain + vco1_amp_mod).clamp(0.0, 1.0)

        # ── VCO 2 (Saw/Sine blend) ────────────────────────────────────────
        vco2_detune = 12.0 * torch.tanh(self.vco2_detune_raw)
        vco2_f = f0_hz * (2.0 ** (vco2_detune / 12.0)) * (1.0 + 0.1 * vco2_pitch_mod)
        vco2_phase = 2.0 * math.pi * torch.cumsum(vco2_f / sr, dim=0)
        vco2_init  = 2.0 * math.pi * torch.sigmoid(self.vco2_phase_raw)
        vco2_sine  = torch.sin(vco2_phase + vco2_init)
        vco2_saw   = _saw_wave(vco2_phase + vco2_init)
        shape      = torch.sigmoid(self.vco2_shape_raw)           # 0=sine, 1=saw
        vco2_raw   = (1.0 - shape) * vco2_sine + shape * vco2_saw
        vco2_gain  = torch.sigmoid(self.vco2_gain_raw)
        vco2_out   = vco2_raw * (vco2_gain + vco2_amp_mod).clamp(0.0, 1.0)

        # ── Noise ─────────────────────────────────────────────────────────
        noise_raw  = self._noise_buf.to(device)
        noise_gain = 0.025 * torch.sigmoid(self.noise_gain_raw)   # matches Voice scaling
        noise_out  = noise_raw * noise_gain * (noise_amp_mod + 0.5).clamp(0.0, 1.0)

        # ── Audio Mixer ───────────────────────────────────────────────────
        w = torch.softmax(self.mixer_raw, dim=0)
        mixed = w[0] * vco1_out + w[1] * vco2_out + w[2] * noise_out

        # Peak-normalize
        peak = mixed.abs().max().clamp(min=1e-8)
        return mixed / peak * 0.9

    def get_param_dict(self) -> dict:
        with torch.no_grad():
            w = torch.softmax(self.mixer_raw, dim=0).tolist()
            return {
                "midi_hz": round(self._kbd_midi_hz().item(), 1),
                "note_on_s": round(self._kbd_note_on().item(), 2),
                "vco1_detune_st": round(12.0 * math.tanh(self.vco1_detune_raw.item()), 2),
                "vco2_detune_st": round(12.0 * math.tanh(self.vco2_detune_raw.item()), 2),
                "vco2_shape": round(torch.sigmoid(self.vco2_shape_raw).item(), 2),
                "lfo1_rate_hz": round(20.0 * torch.sigmoid(self.lfo1_rate_raw).item(), 2),
                "lfo2_rate_hz": round(20.0 * torch.sigmoid(self.lfo2_rate_raw).item(), 2),
                "mix_vco1": round(w[0], 3),
                "mix_vco2": round(w[1], 3),
                "mix_noise": round(w[2], 3),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                   fft_sizes: tuple = (2048, 1024, 512, 256)) -> torch.Tensor:
    """Multi-scale spectral distance (linear + log magnitude), matching the paper."""
    loss = pred.new_zeros(1)
    for n in fft_sizes:
        hop = max(n // 4, 1)
        win = torch.hann_window(n, device=pred.device)
        def S(x, n=n, hop=hop, win=win):
            return torch.stft(
                x, n_fft=n, hop_length=hop, win_length=n, window=win, return_complex=True
            ).abs()
        Sp, St = S(pred), S(target)
        loss = loss + (Sp - St).abs().mean()
        loss = loss + (Sp.clamp(1e-7).log() - St.clamp(1e-7).log()).abs().mean()
    return loss


def fit_synth_to_target(
    target_audio: torch.Tensor,
    n_iter: int = 200,
    lr: float = 0.001,
    sample_rate: int = SAMPLE_RATE,
    verbose: bool = True,
) -> tuple:
    """
    Optimise VoiceSynth parameters to match target_audio.

    Exactly replicates Hagiwara et al. (2022): Adam, 200 iterations, lr=0.001,
    multi-scale STFT loss, uniform-random parameter initialisation.

    Args:
        target_audio: (T,) mono waveform
        n_iter:       number of Adam iterations (200 as in the paper)
        lr:           learning rate (0.001 as in the paper)
        sample_rate:  audio sample rate
        verbose:      print loss every 40 steps

    Returns:
        (synth, loss_history)
    """
    duration = len(target_audio) / sample_rate
    synth = VoiceSynth(sample_rate=sample_rate, duration=duration)

    # Seed per-clip so each animal gets an independent, reproducible initialisation.
    # Without this, sequential calls in the same cell share RNG state and all animals
    # start from nearly the same position, converging to similar local minima.
    seed = int(target_audio[:200].abs().sum().item() * 1e7) % (2 ** 31)
    torch.manual_seed(seed)

    # Uniform random initialisation (as in the paper — not the default init values)
    with torch.no_grad():
        for p in synth.parameters():
            p.uniform_(-1.5, 1.5)

    optimizer = torch.optim.Adam(synth.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(synth.parameters(), lr=lr)
    target_audio = target_audio.detach()
    T = len(target_audio)
    losses = []

    for step in range(n_iter):
        optimizer.zero_grad()
        pred = synth()
        loss = _spectral_loss(pred[:T], target_audio[:T])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (step % 40 == 0 or step == n_iter - 1):
            p = synth.get_param_dict()
            print(f"  step {step:3d}/{n_iter}  loss={loss.item():.3f}  "
                  f"f0={p['midi_hz']:.0f}Hz  vco2={p['vco2_detune_st']:+.1f}st  "
                  f"mix=({p['mix_vco1']:.2f},{p['mix_vco2']:.2f},{p['mix_noise']:.2f})  "
                  f"lfo1={p['lfo1_rate_hz']:.1f}Hz lfo2={p['lfo2_rate_hz']:.1f}Hz")

    return synth, losses


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (ESC-50)
# ─────────────────────────────────────────────────────────────────────────────

def load_esc50_animals(
    esc50_dir: str,
    categories: set | None = None,
    max_per_category: int = 1,
    duration: float = DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """
    Load animal vocalization clips from ESC-50.

    Returns list of {'audio': (T,), 'category': str, 'filename': str}.
    """
    if categories is None:
        categories = {"cat", "frog", "chirping_birds", "crickets"}

    meta_path = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")
    selected: list[dict] = []
    counts: dict[str, int] = {}

    rows = []
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in categories:
                rows.append(row)

    random.shuffle(rows)

    for row in rows:
        cat = row["category"]
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
        waveform = waveform.squeeze(0)

        n_target = int(duration * sample_rate)
        if waveform.shape[0] >= n_target:
            waveform = waveform[:n_target]
        else:
            waveform = F.pad(waveform, (0, n_target - waveform.shape[0]))

        peak = waveform.abs().max().clamp(min=1e-8)
        waveform = waveform / peak * 0.9

        selected.append({"audio": waveform, "category": cat, "filename": row["filename"]})
        counts[cat] = counts.get(cat, 0) + 1

    return selected
