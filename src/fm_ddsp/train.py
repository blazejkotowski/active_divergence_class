"""
Training script for the DDX7-style FM violin model.

Trains DDSPFMModel (6-operator FM with violin "STRINGS 1" DX7 patch) on the
same preprocessed violin clips used by the harmonic DDSP baseline.

Usage:
    python -m src.fm_ddsp.train [--steps N] [--batch-size B] [--lr LR]

The best checkpoint is saved to models/fm_violin.pt — the same format as
ddsp_baseline_violin.pt so notebook cells can swap models with one path change.
"""
import os
import sys
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.fm_ddsp.model import DDSPFMModel
from src.ddsp.loss import MultiScaleSpectralLoss
from src.ddsp.dataset import URMPViolinDataset

PROCESSED_DIR = "/home/btadeusz/code/active_divergence_class/samples/processed"
MODELS_DIR    = "/home/btadeusz/code/active_divergence_class/models"
CHECKPOINT    = os.path.join(MODELS_DIR, "fm_violin.pt")
SAMPLE_RATE   = 16000

FM_CONFIG = {
    "hidden_size":    256,
    "n_mlp_layers":   3,
    "n_noise_bands":  65,
    "I_max":          2.0,
    "reverb_length":  48000,
    "sample_rate":    SAMPLE_RATE,
    "hop_length":     64,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int,   default=5000)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--val-every",  type=int,   default=500)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training FM violin model on {device}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    dataset = URMPViolinDataset(PROCESSED_DIR)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=URMPViolinDataset.collate_fn,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=URMPViolinDataset.collate_fn,
                              num_workers=2, pin_memory=True)

    model     = DDSPFMModel(**FM_CONFIG).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    criterion = MultiScaleSpectralLoss().to(device)

    best_val = float("inf")
    step     = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    print(f"Training for {args.steps} steps...")
    while step < args.steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        audio    = batch["audio"].to(device)
        f0       = batch["f0_hz"].to(device)
        loudness = batch["loudness"].to(device)

        out  = model(f0, loudness)
        loss = criterion(out["audio"], audio)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    vo = model(vb["f0_hz"].to(device), vb["loudness"].to(device))
                    val_losses.append(criterion(vo["audio"], vb["audio"].to(device)).item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"  [val] step {step} | val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": FM_CONFIG,
                }, CHECKPOINT)
                print(f"  ✓ Saved {CHECKPOINT} (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
