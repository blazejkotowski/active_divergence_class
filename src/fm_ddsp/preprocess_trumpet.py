"""
Preprocess URMP trumpet stems into fixed-length training clips.

Usage:
    python -m src.fm_ddsp.preprocess_trumpet
"""
import os
import sys
import torch
import torchaudio
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ddsp.features import extract_loudness, extract_f0

SAMPLE_RATE = 16000
HOP_LENGTH  = 64
CLIP_DURATION = 4.0
CLIP_SAMPLES  = int(CLIP_DURATION * SAMPLE_RATE)

URMP_ROOT    = "/mnt/mariadata/datasets/URMP/Dataset"
SAMPLES_DIR  = "/home/btadeusz/code/active_divergence_class/samples"
OUT_DIR      = os.path.join(SAMPLES_DIR, "trumpet_processed")

MIN_VOICED_FRACTION = 0.25

TRUMPET_STEMS = [
    ("05_Entertainer_tpt_tpt",        "AuSep_1_tpt_05_Entertainer.wav"),
    ("05_Entertainer_tpt_tpt",        "AuSep_2_tpt_05_Entertainer.wav"),
    ("07_GString_tpt_tbn",            "AuSep_1_tpt_07_GString.wav"),
    ("09_Jesus_tpt_vn",               "AuSep_1_tpt_09_Jesus.wav"),
    ("10_March_tpt_sax",              "AuSep_1_tpt_10_March.wav"),
    ("15_Surprise_tpt_tpt_tbn",       "AuSep_1_tpt_15_Surprise.wav"),
    ("15_Surprise_tpt_tpt_tbn",       "AuSep_2_tpt_15_Surprise.wav"),
    ("18_Nocturne_vn_fl_tpt",         "AuSep_3_tpt_18_Nocturne.wav"),
    ("20_Pavane_tpt_vn_vc",           "AuSep_1_tpt_20_Pavane.wav"),
    ("31_Slavonic_tpt_tpt_hn_tbn",    "AuSep_1_tpt_31_Slavonic.wav"),
    ("33_Elise_tpt_tpt_hn_tbn",       "AuSep_1_tpt_33_Elise.wav"),
    ("34_Fugue_tpt_tpt_hn_tbn",       "AuSep_1_tpt_34_Fugue.wav"),
    ("42_Arioso_tpt_tpt_hn_tbn_tba",  "AuSep_1_tpt_42_Arioso.wav"),
    ("43_Chorale_tpt_tpt_hn_tbn_tba", "AuSep_1_tpt_43_Chorale.wav"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clip_idx = 0
    print("Preprocessing URMP trumpet stems...")
    for folder, wav_name in TRUMPET_STEMS:
        wav_path = os.path.join(URMP_ROOT, folder, wav_name)
        if not os.path.exists(wav_path):
            print(f"  SKIP (not found): {wav_path}")
            continue

        print(f"  Loading: {wav_name}")
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        audio_full = waveform.squeeze(0)

        n_clips = len(audio_full) // CLIP_SAMPLES
        n_kept = n_dropped = 0
        for i in tqdm.tqdm(range(n_clips), desc=f"    {wav_name}"):
            clip = audio_full[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES].clone()
            f0_hz, voiced = extract_f0(clip, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device="cuda")
            if voiced.float().mean().item() < MIN_VOICED_FRACTION:
                n_dropped += 1
                continue
            loudness = extract_loudness(clip, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            N = min(f0_hz.shape[0], loudness.shape[0], CLIP_SAMPLES // HOP_LENGTH)
            torch.save({
                "audio": clip[:N * HOP_LENGTH].float(),
                "f0_hz": f0_hz[:N].float(),
                "loudness": loudness[:N].float(),
                "filename": wav_name,
            }, os.path.join(OUT_DIR, f"clip_{clip_idx:04d}.pt"))
            clip_idx += 1
            n_kept += 1
        print(f"    kept {n_kept}, dropped {n_dropped}")

    print(f"\nDone. {clip_idx} clips saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
