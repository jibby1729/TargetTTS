"""
Prepare data for extraction network training.

This script does two things:
  1. Splits the recipe CSV into train/val/test sets (10000/1250/1250)
  2. Pre-computes ECAPA-TDNN speaker embeddings for all enrollment clips

Since the speaker encoder is frozen, we only need to compute embeddings once.
This saves significant time when experimenting with different training configs.

The output is saved to a preparation directory that the training script loads from.

Usage:
    python -m scripts.prepare_training
    python -m scripts.prepare_training --csv data/mixture_recipes/librispeech_extraction_recipes.csv
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.speaker_encoder import SpeakerEncoder


def split_recipes(df, seed=42):
    """
    Shuffle and split recipes into train/val/test.

    Returns three DataFrames: 10000 train, 1250 val, 1250 test.
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = 10000
    n_val = 1250
    n_test = 1250

    if len(df) < n_train + n_val + n_test:
        raise ValueError(
            f"Need at least {n_train + n_val + n_test} recipes, got {len(df)}"
        )

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:n_train + n_val + n_test]

    return df_train, df_val, df_test


def precompute_embeddings(recipes_df, device):
    """
    Compute speaker embeddings for every unique enrollment clip in the recipes.

    Returns a dict mapping enrollment audio path -> embedding tensor (192,).
    """
    encoder = SpeakerEncoder(device=device)
    unique_paths = recipes_df["enrollment_audio_path"].unique()
    print(f"Computing embeddings for {len(unique_paths)} unique enrollment clips...")

    embeddings = {}
    for path in tqdm(unique_paths, desc="  embeddings"):
        embeddings[path] = encoder.encode_file(path)

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare data for extraction training")
    parser.add_argument("--csv", type=str,
                        default="data/mixture_recipes/librispeech_extraction_recipes.csv")
    parser.add_argument("--output-dir", type=str, default="data/prepared_training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Load recipes
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} recipes from {args.csv}")

    # Split into train/val/test
    df_train, df_val, df_test = split_recipes(df, seed=args.seed)
    print(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")
    print()

    # Pre-compute embeddings for train and val sets
    # (test embeddings are computed separately during evaluation)
    df_train_val = pd.concat([df_train, df_val])
    embeddings = precompute_embeddings(df_train_val, device)
    print()

    # Save everything
    os.makedirs(args.output_dir, exist_ok=True)

    df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    torch.save(embeddings, os.path.join(args.output_dir, "embeddings.pt"))

    print(f"Saved splits and embeddings to {args.output_dir}/")
    print(f"  train.csv      ({len(df_train)} rows)")
    print(f"  val.csv        ({len(df_val)} rows)")
    print(f"  test.csv       ({len(df_test)} rows)")
    print(f"  embeddings.pt  ({len(embeddings)} embeddings)")


if __name__ == "__main__":
    main()
