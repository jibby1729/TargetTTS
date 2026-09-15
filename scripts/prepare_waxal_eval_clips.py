"""Download a small WAXAL aka_asr audio subset for local KenLM eval.

Writes 16 kHz wav + reference text under a gitignored directory.
Does not touch train audio except when --split validation|test.

Usage:
    python3 -m scripts.prepare_waxal_eval_clips \\
      --split validation --n 5 --out data/kenlm/eval/val --seed 1508
    python3 -m scripts.prepare_waxal_eval_clips \\
      --split test --n 30 --out data/kenlm/eval/test --seed 1508
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from src.preprocess import normalize_akan_text

HF_DATASET = "google/WaxalNLP"
HF_CONFIG = "aka_asr"
ENGLISH_POLICY_VERSION = "waxal-aka-reference-hints-v1"
ENGLISH_MIN_HITS = 2

# Skip refs that look like mixed Twi/English image descriptions.
ENGLISH_HINTS = {
    "meeting",
    "table",
    "computer",
    "blue",
    "helmet",
    "zebra",
    "crossing",
    "phone",
    "books",
    "book",
    "mic",
    "rubber",
    "colour",
    "color",
    "television",
    "coat",
    "beans",
    "bowl",
    "round",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a fixed-seed WAXAL aka_asr clip subset."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
        help="WAXAL aka_asr split (never train)",
    )
    parser.add_argument("--n", type=int, required=True, help="How many clips to keep")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=1508)
    parser.add_argument(
        "--pool",
        type=int,
        default=None,
        help="How many source rows to inspect before sampling (default: max(120, 4*n))",
    )
    parser.add_argument(
        "--filter-english",
        action="store_true",
        help="Drop refs with two or more English hint words before sampling",
    )
    return parser.parse_args()


def english_hits(text: str) -> list[str]:
    tokens = set(normalize_akan_text(text).split())
    return sorted(word for word in ENGLISH_HINTS if word in tokens)


def is_english_heavy(text: str, min_hits: int = ENGLISH_MIN_HITS) -> bool:
    return len(english_hits(text)) >= min_hits


def _audio_bytes(audio) -> tuple[bytes, str]:
    if audio is None:
        raise ValueError("missing audio")
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        path = audio.get("path") or "clip.mp3"
        if raw:
            return bytes(raw), str(path)
        if path and Path(path).is_file():
            return Path(path).read_bytes(), str(path)
    raise ValueError(f"unsupported audio payload: {type(audio)}")


def write_16k_wav(raw: bytes, src_name: str, dest_wav: Path) -> None:
    suffix = Path(src_name).suffix or ".mp3"
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        src_path = handle.name
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                str(dest_wav),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dest_wav.is_file():
            raise RuntimeError(
                f"ffmpeg failed for {src_name}: {proc.stderr[-500:]}"
            )
    finally:
        Path(src_path).unlink(missing_ok=True)


def iter_rows(split: str):
    from datasets import Audio, load_dataset

    ds = load_dataset(
        HF_DATASET,
        HF_CONFIG,
        split=split,
        streaming=True,
    )
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"cast_column Audio(decode=False) failed ({exc}); trying as-is", flush=True)
    for row in ds:
        yield row


def collect_pool(
    split: str,
    pool: int,
    filter_english: bool = False,
) -> tuple[list[dict], list[dict], int]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    inspected = 0
    print(f"Streaming {HF_CONFIG} split={split} ({pool} source rows)...", flush=True)
    for source_index, row in enumerate(iter_rows(split)):
        if source_index >= pool:
            break
        inspected += 1
        raw_text = row.get("transcription")
        audio = row.get("audio")
        src_name = (
            str(audio.get("path") or "unknown")
            if isinstance(audio, dict)
            else "unknown"
        )
        if not isinstance(raw_text, str) or not raw_text.strip():
            rejected.append(
                {
                    "source_index": source_index,
                    "src_name": src_name,
                    "reason": "missing_transcription",
                    "english_hits": [],
                }
            )
            continue
        hits = english_hits(raw_text)
        if filter_english and len(hits) >= ENGLISH_MIN_HITS:
            rejected.append(
                {
                    "source_index": source_index,
                    "src_name": src_name,
                    "reason": "english_hint_threshold",
                    "english_hits": hits,
                    "reference": raw_text,
                }
            )
            continue
        try:
            audio_bytes, src_name = _audio_bytes(audio)
        except Exception as exc:
            rejected.append(
                {
                    "source_index": source_index,
                    "src_name": src_name,
                    "reason": "audio_unavailable",
                    "detail": str(exc),
                    "english_hits": hits,
                    "reference": raw_text,
                }
            )
            continue
        if not audio_bytes:
            rejected.append(
                {
                    "source_index": source_index,
                    "src_name": src_name,
                    "reason": "empty_audio",
                    "english_hits": hits,
                    "reference": raw_text,
                }
            )
            continue
        accepted.append(
            {
                "source_index": source_index,
                "transcription": raw_text,
                "audio_bytes": audio_bytes,
                "src_name": src_name,
                "english_hits": hits,
            }
        )
        if inspected % 10 == 0:
            print(
                f"  ... inspected {inspected}; eligible={len(accepted)}; "
                f"rejected={len(rejected)}",
                flush=True,
            )
    if filter_english:
        n_english = sum(
            row["reason"] == "english_hint_threshold" for row in rejected
        )
        print(f"  rejected {n_english} English-heavy refs", flush=True)
    return accepted, rejected, inspected


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("error: --n must be >= 1")
    pool = args.pool if args.pool is not None else max(120, 4 * args.n)
    pool = max(pool, args.n)

    rows, rejected, inspected = collect_pool(
        args.split,
        pool,
        filter_english=args.filter_english,
    )
    if len(rows) < args.n:
        raise SystemExit(
            f"error: only collected {len(rows)} usable {args.split} rows; need {args.n}"
        )

    rng = random.Random(args.seed)
    chosen = rng.sample(rows, args.n)
    chosen_source_indices = {row["source_index"] for row in chosen}
    args.out.mkdir(parents=True, exist_ok=True)

    accepted_manifest = []
    for i, row in enumerate(chosen):
        clip_id = f"{i:03d}"
        wav_path = args.out / f"{clip_id}.wav"
        ref_path = args.out / f"{clip_id}.ref.txt"
        print(f"Writing {wav_path.name} from {row['src_name']}", flush=True)
        write_16k_wav(row["audio_bytes"], row["src_name"], wav_path)
        ref_path.write_text(row["transcription"].strip() + "\n", encoding="utf-8")
        accepted_manifest.append(
            {
                "id": clip_id,
                "wav": wav_path.name,
                "ref": ref_path.name,
                "split": args.split,
                "source_index": row["source_index"],
                "src_name": row["src_name"],
                "english_hits": row.get("english_hits", []),
            }
        )

    manifest = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": args.split,
        "seed": args.seed,
        "requested_clips": args.n,
        "source_rows_requested": pool,
        "source_rows_inspected": inspected,
        "eligible_rows": len(rows),
        "selection_before_decoding": True,
        "policy": {
            "version": ENGLISH_POLICY_VERSION,
            "reference_only": True,
            "filter_enabled": args.filter_english,
            "reject_at_least_n_hits": ENGLISH_MIN_HITS,
            "english_hints": sorted(ENGLISH_HINTS),
        },
        "accepted": accepted_manifest,
        "rejected": rejected,
        "eligible_not_selected": [
            {
                "source_index": row["source_index"],
                "src_name": row["src_name"],
                "english_hits": row["english_hits"],
            }
            for row in rows
            if row["source_index"] not in chosen_source_indices
        ],
    }
    man_path = args.out / "manifest.json"
    man_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(accepted_manifest)} clips to {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
