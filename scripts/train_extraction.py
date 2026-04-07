"""
Train the speaker-conditioned extraction network.

Loads prepared data (splits + embeddings) from prepare_training.py,
then trains the mask network with MSE loss on spectrograms.

Usage:
    python -m scripts.train_extraction
    python -m scripts.train_extraction --epochs 50 --batch-size 8 --lr 1e-3
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.extraction_net import ExtractionNet
from src.dataset import MixtureDataset, pad_collate


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mix_mag, target_mag, speaker_emb in tqdm(loader, desc="  train", leave=False):
        mix_mag = mix_mag.to(device)
        target_mag = target_mag.to(device)
        speaker_emb = speaker_emb.to(device)

        mask = model(mix_mag, speaker_emb)
        estimated = mask * mix_mag
        loss = nn.functional.mse_loss(estimated, target_mag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for mix_mag, target_mag, speaker_emb in tqdm(loader, desc="  val  ", leave=False):
        mix_mag = mix_mag.to(device)
        target_mag = target_mag.to(device)
        speaker_emb = speaker_emb.to(device)

        mask = model(mix_mag, speaker_emb)
        estimated = mask * mix_mag
        loss = nn.functional.mse_loss(estimated, target_mag)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train the extraction network")
    parser.add_argument("--prepared-dir", type=str, default="data/prepared_training")
    parser.add_argument("--mix-dir", type=str,
                        default="data/synthetic_mixtures/librispeech_extraction")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}")
    print()

    # Load prepared data
    df_train = pd.read_csv(os.path.join(args.prepared_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.prepared_dir, "val.csv"))
    embeddings = torch.load(
        os.path.join(args.prepared_dir, "embeddings.pt"), weights_only=False
    )
    print(f"Loaded {len(df_train)} train / {len(df_val)} val recipes")
    print(f"Loaded {len(embeddings)} pre-computed embeddings")
    print()

    # Dataloaders
    train_loader = DataLoader(
        MixtureDataset(df_train, args.mix_dir, embeddings),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=pad_collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        MixtureDataset(df_val, args.mix_dir, embeddings),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_collate, num_workers=args.num_workers, pin_memory=True,
    )

    # Model, optimizer, scheduler
    model = ExtractionNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    print()
    header = f"{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}  {'Best':>4}  {'Time':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_path)
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - t0
        flag = "*" if is_best else ""
        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_loss:>12.6f}  {lr:>10.1e}  {flag:>4}  {elapsed:>5.1f}s")

        if epochs_without_improvement >= args.patience:
            print(f"\nStopping early at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
