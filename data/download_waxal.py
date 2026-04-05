"""
Download the WAXAL (AkanASR) dataset from HuggingFace Hub and save to disk.

Usage:
    uv run python data/download_waxal.py [--root DATA_DIR] [--splits SPLIT [SPLIT ...]]

Example:
    uv run python data/download_waxal.py --root ./data/waxal --splits test
"""

import argparse
from pathlib import Path
from datasets import load_dataset

AVAILABLE_SPLITS = ["train", "validation", "test"]
DEFAULT_SPLITS = ["test"]


def print_split_stats(ds, split):
    """Print basic statistics for a dataset split."""
    speakers = set(ds["speaker_id"])
    print(f"  Stats: {len(ds)} utterances, {len(speakers)} speakers")


def main():
    parser = argparse.ArgumentParser(description="Download WAXAL AkanASR dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="./data/waxal",
        help="Root directory to store downloaded data",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=AVAILABLE_SPLITS,
        help=f"Splits to download (default: {DEFAULT_SPLITS})",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {root.resolve()}")
    print(f"Splits: {args.splits}\n")

    print("Downloading WaxalNLP AkanASR dataset from HuggingFace Hub...")
    ds = load_dataset("google/WaxalNLP", "aka_asr")

    # Drop unlabeled split — no transcriptions
    del ds["unlabeled"]

    # Filter to requested splits
    for split in list(ds.keys()):
        if split not in args.splits:
            del ds[split]

    ds.save_to_disk(str(root))
    print(f"Saved to {root}")

    for split in args.splits:
        print(f"\nSplit: {split}")
        print_split_stats(ds[split], split)

    print("\nDone.")


if __name__ == "__main__":
    main()