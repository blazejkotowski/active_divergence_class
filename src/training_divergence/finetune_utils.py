"""
Fine-tuning utilities for training-time active divergence (Sections 3.2 and 3.3).

These helpers are imported by the notebook; the training loop itself
lives in the notebook so students can read and modify it directly.

Key design decisions:
- load_audio_chunks: handles arbitrary WAV directories (not .pt preprocessed clips)
- extract_features_from_audio: thin wrapper around the existing DDSP feature extractors
- finetune_step: single gradient step; returns loss scalar
- make_probe_trajectory: fixed (f0, loudness) probe for comparing model snapshots
- save_divergent_checkpoint: keeps naming convention models/ddsp_divergent_<source>_<target>_step<N>.pt
"""

import os
import sys
import pathlib
from typing import Optional

import torch
import torchaudio

ROOT = pathlib.Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ddsp.features import extract_loudness, extract_f0
from src.ddsp.model import DDSPAutoencoder

SAMPLE_RATE = 16000
HOP_LENGTH = 64


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_chunks(
    audio_dir: str,
    chunk_duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
    max_chunks: int = 30,
    fixed_f0_hz: Optional[float] = None,
    hop_length: int = HOP_LENGTH,
) -> list[dict]:
    """
    Load WAV files from ``audio_dir`` and slice into fixed-length chunks.

    Each returned dict has:
        'audio':    (T,) float32 mono waveform at ``sample_rate``
        'f0_hz':    (N_frames,) fundamental frequency
        'loudness': (N_frames,) A-weighted log loudness

    Args:
        audio_dir:      directory containing .wav files
        chunk_duration: seconds per chunk
        sample_rate:    target sample rate
        max_chunks:     stop after this many chunks (for CPU-feasible training)
        fixed_f0_hz:    if set, skip F0 extraction and use this constant instead.
                        Useful for unpitched material (e.g. EMF noise) where
                        CREPE returns garbage.
        hop_length:     STFT hop in samples (must match the model)
    """
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []

    wav_files = sorted([
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    ])
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    for fname in wav_files:
        if len(chunks) >= max_chunks:
            break
        path = os.path.join(audio_dir, fname)
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.squeeze(0)  # (T_full,)

        # Split into non-overlapping chunks
        n_full = waveform.shape[0] // chunk_samples
        for i in range(n_full):
            if len(chunks) >= max_chunks:
                break
            audio = waveform[i * chunk_samples: (i + 1) * chunk_samples].clone()

            # Peak-normalise
            peak = audio.abs().max().clamp(min=1e-8)
            audio = audio / peak * 0.9

            loudness = extract_loudness(audio, sr=sample_rate, hop_length=hop_length)

            if fixed_f0_hz is not None:
                n_frames = audio.shape[0] // hop_length
                f0_hz = torch.full((n_frames,), fixed_f0_hz)
            else:
                f0_hz = extract_f0(audio, sr=sample_rate, hop_length=hop_length)

            # Align lengths (feature extraction may produce slightly different counts)
            n = min(loudness.shape[0], f0_hz.shape[0])
            chunks.append({
                "audio": audio,
                "f0_hz": f0_hz[:n],
                "loudness": loudness[:n],
            })

    return chunks


def extract_features_from_audio(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    fixed_f0_hz: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (f0_hz, loudness) from a (T,) mono waveform.

    Convenience wrapper used for one-off clips (e.g. the probe trajectory).
    """
    loudness = extract_loudness(audio, sr=sample_rate, hop_length=hop_length)
    if fixed_f0_hz is not None:
        n_frames = audio.shape[0] // hop_length
        f0_hz = torch.full((n_frames,), fixed_f0_hz)
    else:
        f0_hz = extract_f0(audio, sr=sample_rate, hop_length=hop_length)
    n = min(loudness.shape[0], f0_hz.shape[0])
    return f0_hz[:n], loudness[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Probe trajectory
# ─────────────────────────────────────────────────────────────────────────────

def make_probe_trajectory(
    f0_hz: float = 220.0,
    loudness_db: float = -30.0,
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constant-pitch, constant-loudness probe trajectory for tracking model drift.

    Returns (f0, loudness) tensors shaped (1, N_frames) ready for model().
    Render the same probe at every checkpoint to hear how the model changes.

    Args:
        f0_hz:       fundamental frequency (Hz)
        loudness_db: loudness (dB, A-weighted; roughly -60 = quiet, -20 = loud)
        duration:    probe duration (seconds)
    """
    n_frames = int(duration * sample_rate / hop_length)
    f0 = torch.full((1, n_frames), f0_hz)
    loudness = torch.full((1, n_frames), loudness_db)
    return f0, loudness


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def finetune_step(
    model: DDSPAutoencoder,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    batch: dict,
    device: str = "cpu",
) -> float:
    """
    Execute one gradient step of fine-tuning.

    Args:
        model:     DDSPAutoencoder (weights unfrozen)
        optimizer: e.g. Adam
        loss_fn:   MultiScaleSpectralLoss instance
        batch:     dict with keys 'audio', 'f0_hz', 'loudness' — all (B, *) tensors
        device:    'cpu' or 'cuda'

    Returns:
        scalar loss value
    """
    model.train()
    audio = batch["audio"].to(device)        # (B, T)
    f0 = batch["f0_hz"].to(device)           # (B, N)
    loudness = batch["loudness"].to(device)  # (B, N)

    out = model(f0, loudness)
    loss = loss_fn(out["audio"], audio)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

def save_divergent_checkpoint(
    model: DDSPAutoencoder,
    step: int,
    source_tag: str,
    target_tag: str,
    models_dir: str,
    config: dict,
) -> str:
    """
    Save a divergent checkpoint as ``models/ddsp_divergent_<source>_<target>_step<N>.pt``.

    Does not overwrite the baseline checkpoint.

    Returns the saved file path.
    """
    os.makedirs(models_dir, exist_ok=True)
    fname = f"ddsp_divergent_{source_tag}_{target_tag}_step{step:04d}.pt"
    path = os.path.join(models_dir, fname)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "source_tag": source_tag,
        "target_tag": target_tag,
        "config": config,
    }, path)
    return path


def save_inharmonic_checkpoint(
    model: DDSPAutoencoder,
    step: int,
    weight: float,
    models_dir: str,
    config: dict,
) -> str:
    """
    Save an inharmonicity checkpoint as ``models/ddsp_inharmonic_w<W>_step<N>.pt``.

    Returns the saved file path.
    """
    os.makedirs(models_dir, exist_ok=True)
    w_str = f"{weight:.3f}".replace(".", "p")
    fname = f"ddsp_inharmonic_w{w_str}_step{step:04d}.pt"
    path = os.path.join(models_dir, fname)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "inharmonicity_weight": weight,
        "config": config,
    }, path)
    return path
