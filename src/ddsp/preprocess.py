"""
Preprocess URMP violin stems into fixed-length training clips.

Usage:
    python -m src.ddsp.preprocess
"""
import os
import sys
import glob
import torch
import torchaudio
import tqdm

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ddsp.features import extract_loudness, extract_f0

SAMPLE_RATE = 16000
HOP_LENGTH = 64
CLIP_DURATION = 4.0
CLIP_SAMPLES = int(CLIP_DURATION * SAMPLE_RATE)
CLIP_FRAMES = CLIP_SAMPLES // HOP_LENGTH

URMP_ROOT = "/mnt/mariadata/datasets/URMP/Dataset"
SAMPLES_DIR = "/home/btadeusz/code/active_divergence_class/samples"
PROCESSED_DIR = os.path.join(SAMPLES_DIR, "processed")

# Drop clips with fewer than this fraction of voiced (non-silent) frames.
# Catches lead-in/trail-out silence and rests without hardcoding a skip duration.
MIN_VOICED_FRACTION = 0.25

# Violin stems to use (selected for solo melodic content and diversity)
VIOLIN_STEMS = [
    # (folder, stem_filename, f0_annotation_filename)
    ("09_Jesus_tpt_vn", "AuSep_2_vn_09_Jesus.wav", "F0s_2_vn_09_Jesus.txt"),
    ("17_Nocturne_vn_fl_cl", "AuSep_1_vn_17_Nocturne.wav", "F0s_1_vn_17_Nocturne.txt"),
    ("26_King_vn_vn_va_vc", "AuSep_1_vn_26_King.wav", "F0s_1_vn_26_King.txt"),
    ("44_K515_vn_vn_va_va_vc", "AuSep_1_vn_44_K515.wav", "F0s_1_vn_44_K515.txt"),
]


def load_and_resample(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)  # (T,)


def slice_clips(audio: torch.Tensor) -> list[torch.Tensor]:
    clips = []
    n_clips = len(audio) // CLIP_SAMPLES
    for i in range(n_clips):
        clips.append(audio[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES])
    return clips


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    clip_idx = 0
    raw_copies = []

    print("Preprocessing URMP violin stems...")
    for folder, wav_name, _ in VIOLIN_STEMS:
        wav_path = os.path.join(URMP_ROOT, folder, wav_name)
        if not os.path.exists(wav_path):
            print(f"  SKIP (not found): {wav_path}")
            continue

        print(f"  Loading: {wav_path}")
        audio = load_and_resample(wav_path)
        clips = slice_clips(audio)
        print(f"    → {len(clips)} clips of {CLIP_DURATION}s at {SAMPLE_RATE}Hz")

        # Save a raw copy to samples/ for notebook demos
        raw_out = os.path.join(SAMPLES_DIR, wav_name.replace(".wav", "_16k.wav"))
        torchaudio.save(raw_out, audio.unsqueeze(0), SAMPLE_RATE)
        raw_copies.append(raw_out)

        n_kept = 0
        n_dropped = 0
        for clip in tqdm.tqdm(clips, desc=f"    Extracting features ({wav_name})"):
            # F0 first — we need voiced mask for the silence filter
            f0_hz, voiced = extract_f0(clip, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device="cuda")

            voiced_frac = voiced.float().mean().item()
            if voiced_frac < MIN_VOICED_FRACTION:
                n_dropped += 1
                continue  # skip near-silent clips

            loudness = extract_loudness(clip, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

            T = clip.shape[0]
            N = T // HOP_LENGTH
            min_frames = min(loudness.shape[0], f0_hz.shape[0], N)

            item = {
                "audio": clip[:min_frames * HOP_LENGTH].float(),
                "f0_hz": f0_hz[:min_frames].float(),
                "loudness": loudness[:min_frames].float(),
                "filename": wav_name,
            }

            out_path = os.path.join(PROCESSED_DIR, f"clip_{clip_idx:04d}.pt")
            torch.save(item, out_path)
            clip_idx += 1
            n_kept += 1

        print(f"    kept {n_kept}, dropped {n_dropped} silent clips")

    print(f"\nDone. {clip_idx} clips saved to {PROCESSED_DIR}")
    print(f"Raw 16kHz copies: {raw_copies}")


if __name__ == "__main__":
    main()
