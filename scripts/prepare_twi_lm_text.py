"""
Build a clean Twi text file from WAXAL Akan ASR transcripts for KenLM.

Reads only the transcription column from the requested aka_asr splits.
Does not download test/unlabeled splits or decode audio.

Usage:
    uv run python -m scripts.prepare_twi_lm_text
    python3 -m scripts.prepare_twi_lm_text \
      --splits train --output data/kenlm/waxal_aka_train.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocess import normalize_akan_text

HF_DATASET = "google/WaxalNLP"
HF_CONFIG = "aka_asr"
DEFAULT_SPLITS = ("train", "validation")
ALLOWED_SPLITS = ("train", "validation")
DEFAULT_OUTPUT = Path("data/kenlm/waxal_aka_train_val.txt")


def _iter_transcriptions_datasets(split: str):
    """Stream transcriptions via datasets, requesting only that column."""
    from datasets import load_dataset

    ds = load_dataset(
        HF_DATASET,
        HF_CONFIG,
        split=split,
        streaming=True,
        columns=["transcription"],
    )
    for row in ds:
        yield row["transcription"]


def _iter_transcriptions_pyarrow(split: str):
    """Read only the transcription column from remote parquet shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    prefix = f"datasets/{HF_DATASET}/data/ASR/aka/aka-{split}-"
    paths = [
        p
        for p in fs.glob(f"datasets/{HF_DATASET}/data/ASR/aka/*.parquet")
        if p.startswith(prefix) and "unlabeled" not in p
    ]
    if not paths:
        raise FileNotFoundError(
            f"No parquet shards found for {HF_CONFIG} split={split}"
        )
    paths = sorted(paths)
    print(f"  pyarrow: {len(paths)} shard(s) for split={split}", file=sys.stderr)
    for path in paths:
        with fs.open(path, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=["transcription"])
        for value in table.column("transcription").to_pylist():
            yield value


def iter_transcriptions(split: str):
    """Yield raw transcription strings for one split, text-only."""
    try:
        yield from _iter_transcriptions_datasets(split)
        return
    except TypeError as exc:
        print(
            f"datasets columns= not supported ({exc}); "
            "falling back to pyarrow column projection",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"datasets streaming failed ({exc}); "
            "falling back to pyarrow column projection",
            file=sys.stderr,
        )
    yield from _iter_transcriptions_pyarrow(split)


def collect_review_examples(raw: str, cleaned: str, examples: dict[str, tuple[str, str]]):
    """Keep a few raw→cleaned pairs for the post-run review."""
    if "open_e" not in examples and ("ɛ" in raw or "Ɛ" in raw):
        examples["open_e"] = (raw, cleaned)
    if "open_o" not in examples and ("ɔ" in raw or "Ɔ" in raw):
        examples["open_o"] = (raw, cleaned)
    if "punct" not in examples and any(ch in raw for ch in ".,!?;:'‘’"):
        examples["punct"] = (raw, cleaned)
    lowered = raw.lower()
    if "number_word" not in examples and any(
        token in lowered.split() for token in ("baako", "mmienu", "ebaasa", "ɔha")
    ):
        examples["number_word"] = (raw, cleaned)


def prepare(output_path: Path, splits: tuple[str, ...] = DEFAULT_SPLITS) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    ordered: list[str] = []
    split_stats: dict[str, dict[str, int]] = {}
    examples: dict[str, tuple[str, str]] = {}

    for split in splits:
        raw_count = 0
        kept_count = 0
        print(f"Reading {HF_CONFIG} split={split} (transcription only)...", flush=True)
        for raw in iter_transcriptions(split):
            raw_count += 1
            if raw_count % 1000 == 0:
                print(f"  ... {split} {raw_count} rows", flush=True)
            cleaned = normalize_akan_text(raw if isinstance(raw, str) else "")
            if not cleaned:
                continue
            kept_count += 1
            collect_review_examples(str(raw), cleaned, examples)
            if cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        split_stats[split] = {"raw": raw_count, "kept": kept_count}
        print(
            f"  {split}: raw={raw_count} kept={kept_count} "
            f"(non-empty after normalize)",
            flush=True,
        )

    output_path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
    size_bytes = output_path.stat().st_size

    print()
    print(f"Wrote {len(ordered)} unique lines to {output_path}")
    print(f"File size: {size_bytes} bytes ({size_bytes / (1024 * 1024):.2f} MB)")
    print("Per-split counts:")
    for split, stats in split_stats.items():
        print(f"  {split}: raw={stats['raw']} kept={stats['kept']}")
    print(f"  combined unique: {len(ordered)}")

    if examples:
        print("\nSample raw → cleaned pairs:")
        for label, (raw, cleaned) in examples.items():
            print(f"  [{label}]")
            print(f"    raw:     {raw[:240]}")
            print(f"    cleaned: {cleaned[:240]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare clean WAXAL Akan ASR text for KenLM (text-only)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output text file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALLOWED_SPLITS,
        default=list(DEFAULT_SPLITS),
        help="ASR text splits to include (default: train validation)",
    )
    args = parser.parse_args()
    prepare(args.output, tuple(args.splits))


if __name__ == "__main__":
    main()
