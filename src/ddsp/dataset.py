"""Dataset: pre-processed URMP violin stems with precomputed features."""
import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
HOP_LENGTH = 64
CLIP_DURATION = 4.0  # seconds per training example
CLIP_SAMPLES = int(CLIP_DURATION * SAMPLE_RATE)
CLIP_FRAMES = CLIP_SAMPLES // HOP_LENGTH


class URMPViolinDataset(Dataset):
    """
    Loads pre-processed violin clips from a directory of .pt files.

    Each .pt file contains a dict with keys:
        "audio":    (T,) float32 waveform at 16kHz
        "f0_hz":    (N_frames,) float32
        "loudness": (N_frames,) float32
        "filename": str
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".pt")]
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
            "audio": torch.stack([b["audio"] for b in batch]),
            "f0_hz": torch.stack([b["f0_hz"] for b in batch]),
            "loudness": torch.stack([b["loudness"] for b in batch]),
        }
