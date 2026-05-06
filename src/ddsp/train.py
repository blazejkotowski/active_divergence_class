"""
Training script for the DDSP baseline model.

Usage:
    python -m src.ddsp.train [--steps N] [--batch-size B] [--lr LR]
"""
import os
import sys
import argparse
import time
import torch
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ddsp.model import DDSPAutoencoder
from src.ddsp.loss import MultiScaleSpectralLoss
from src.ddsp.dataset import URMPViolinDataset

PROCESSED_DIR = "/home/btadeusz/code/active_divergence_class/samples/processed"
MODELS_DIR = "/home/btadeusz/code/active_divergence_class/models"
TRAINING_DIR = "/home/btadeusz/code/active_divergence_class/training"
CHECKPOINT = os.path.join(MODELS_DIR, "ddsp_baseline_violin.pt")

SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    # Dataset
    dataset = URMPViolinDataset(PROCESSED_DIR)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=URMPViolinDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=URMPViolinDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = DDSPAutoencoder(
        n_harmonics=100,
        n_noise_bands=65,
        hidden_size=512,
        n_mlp_layers=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing decays LR to 1e-5 over the full run — keeps learning alive longer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    criterion = MultiScaleSpectralLoss().to(device)

    writer = SummaryWriter(log_dir=TRAINING_DIR)

    best_val_loss = float("inf")
    step = 0
    train_iter = iter(train_loader)

    log_path = os.path.join(TRAINING_DIR, "train_log.txt")
    log_f = open(log_path, "w")
    log_f.write("step,train_loss,val_loss,lr\n")

    t0 = time.time()
    print(f"Training for {args.steps} steps...")

    while step < args.steps:
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        audio = batch["audio"].to(device)       # (B, T)
        f0 = batch["f0_hz"].to(device)          # (B, N)
        loudness = batch["loudness"].to(device)  # (B, N)

        out = model(f0, loudness)
        loss = criterion(out["audio"], audio)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  step {step}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    va = vbatch["audio"].to(device)
                    vf = vbatch["f0_hz"].to(device)
                    vl = vbatch["loudness"].to(device)
                    vout = model(vf, vl)
                    val_losses.append(criterion(vout["audio"], va).item())

            val_loss = sum(val_losses) / len(val_losses)
            train_loss = loss.item()
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("lr", lr, step)

            # Log one audio reconstruction
            with torch.no_grad():
                sample = model(vf[:1], vl[:1])
                writer.add_audio("audio/original", va[:1], step, sample_rate=SAMPLE_RATE)
                writer.add_audio("audio/reconstructed", sample["audio"][:1], step, sample_rate=SAMPLE_RATE)

            log_f.write(f"{step},{train_loss:.6f},{val_loss:.6f},{lr:.8f}\n")
            log_f.flush()
            print(f"  [val] step {step} | val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "n_harmonics": 100,
                        "n_noise_bands": 65,
                        "hidden_size": 512,
                        "n_mlp_layers": 3,
                        "sample_rate": SAMPLE_RATE,
                        "hop_length": 64,
                        "reverb_ir_length": 64000,
                    },
                }, CHECKPOINT)
                print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    log_f.close()
    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
