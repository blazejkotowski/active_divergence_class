"""Dataset: audio clips with precomputed F0 and loudness features."""
import os
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
HOP_LENGTH = 64
CLIP_DURATION = 4.0  # seconds per training example
CLIP_SAMPLES = int(CLIP_DURATION * SAMPLE_RATE)
CLIP_FRAMES = CLIP_SAMPLES // HOP_LENGTH


class WavClipDataset(Dataset):
    """
    Loads audio clips from a directory of .wav files with companion feature files.

    Each clip consists of:
        clip_NNNN.wav           — (T,) mono float32 waveform at 16 kHz
        clip_NNNN_features.pt   — {'f0_hz': (N_frames,), 'loudness': (N_frames,)}

    This is the primary dataset class for sharing; WAV files are human-listenable
    and the companion feature files are small (~8 KB each).
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.wav_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        )
        assert len(self.wav_files) > 0, f"No .wav files found in {data_dir}"

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int) -> dict:
        wav_name = self.wav_files[idx]
        wav_path  = os.path.join(self.data_dir, wav_name)
        feat_path = os.path.join(
            self.data_dir, wav_name.replace(".wav", "_features.pt")
        )
        audio, _ = torchaudio.load(wav_path)
        audio = audio.squeeze(0)  # (T,)
        features = torch.load(feat_path, weights_only=True)
        return {
            "audio":    audio,
            "f0_hz":    features["f0_hz"],
            "loudness": features["loudness"],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "audio":    torch.stack([b["audio"] for b in batch]),
            "f0_hz":    torch.stack([b["f0_hz"] for b in batch]),
            "loudness": torch.stack([b["loudness"] for b in batch]),
        }


class URMPViolinDataset(Dataset):
    """
    Legacy dataset that loads pre-processed clips from .pt files.
    Kept for backwards compatibility with existing checkpoints and scripts.
    Prefer WavClipDataset for new work.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".pt")
             and not f.endswith("_features.pt")]
        )
        assert len(self.files) > 0, f"No .pt files found in {data_dir}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.data_dir, self.files[idx])
        item = torch.load(path, weights_only=True)
        return item

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "audio":    torch.stack([b["audio"] for b in batch]),
            "f0_hz":    torch.stack([b["f0_hz"] for b in batch]),
            "loudness": torch.stack([b["loudness"] for b in batch]),
        }
