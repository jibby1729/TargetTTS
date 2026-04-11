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


def power_loss(estimated, target, power):
    eps = 1e-8
    return nn.functional.mse_loss((estimated + eps).pow(power), (target + eps).pow(power))


def compute_loss(mask, mix_mag, target_mag, power, hybrid_alpha=None):
    oracle_mask = torch.clamp(target_mag / (mix_mag + 1e-8), min=0.0, max=1.0)
    recon_loss = power_loss(mask * mix_mag, target_mag, power)
    if hybrid_alpha is not None:
        mask_loss = nn.functional.mse_loss(mask, oracle_mask)
        return hybrid_alpha * mask_loss + (1 - hybrid_alpha) * recon_loss
    return recon_loss


def train_one_epoch(model, loader, optimizer, device, power, hybrid_alpha=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mix_mag, target_mag, speaker_emb in tqdm(loader, desc="  train", leave=False):
        mix_mag = mix_mag.to(device)
        target_mag = target_mag.to(device)
        speaker_emb = speaker_emb.to(device)

        mask = model(mix_mag, speaker_emb)
        loss = compute_loss(mask, mix_mag, target_mag, power, hybrid_alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, power, hybrid_alpha=None):
    """Returns (primary_loss, recon_power_loss, mask_mse) averaged over batches."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_mask = 0.0
    n_batches = 0

    for mix_mag, target_mag, speaker_emb in tqdm(loader, desc="  val  ", leave=False):
        mix_mag = mix_mag.to(device)
        target_mag = target_mag.to(device)
        speaker_emb = speaker_emb.to(device)

        mask = model(mix_mag, speaker_emb)
        loss = compute_loss(mask, mix_mag, target_mag, power, hybrid_alpha)
        total_loss += loss.item()

        # Dual metrics: always compute both
        total_recon += power_loss(mask * mix_mag, target_mag, power).item()
        oracle_mask = torch.clamp(target_mag / (mix_mag + 1e-8), min=0.0, max=1.0)
        total_mask += nn.functional.mse_loss(mask, oracle_mask).item()

        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_mask / n_batches


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
    parser.add_argument("--power", type=float, default=0.3,
                        help="Power-law compression exponent for the loss (default: 0.3)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.1,
                        help="Hybrid loss weight: alpha * mask_MSE + (1-alpha) * power_loss (default: 0.1)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}, power {args.power}")
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
    model = ExtractionNet(input_power=args.power).to(device)
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
    header = f"{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Val Recon':>12}  {'Val MaskMSE':>12}  {'LR':>10}  {'Best':>4}  {'Time':>6}"
    print(header)
    print("-" * len(header))

    # Epoch 0: evaluate randomly initialized model
    val_loss_0, val_recon_0, val_mask_0 = validate(model, val_loader, device, args.power, args.hybrid_alpha)
    best_val_loss = val_loss_0
    best_epoch = 0
    print(f"    0  {'--':>12}  {val_loss_0:>12.6f}  {val_recon_0:>12.6f}  {val_mask_0:>12.6f}  {optimizer.param_groups[0]['lr']:>10.1e}  {'*':>4}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.power, args.hybrid_alpha)
        val_loss, val_recon, val_mask = validate(model, val_loader, device, args.power, args.hybrid_alpha)

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
        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_loss:>12.6f}  {val_recon:>12.6f}  {val_mask:>12.6f}  {lr:>10.1e}  {flag:>4}  {elapsed:>5.1f}s")

        if epochs_without_improvement >= args.patience:
            print(f"\nStopping early at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
