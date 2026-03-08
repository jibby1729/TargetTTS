"""
Download and preprocess LibriSpeech dataset splits.

Usage:
    python data/download_librispeech.py [--root DATA_DIR] [--splits SPLIT [SPLIT ...]]

Example:
    python data/download_librispeech.py --root ./data/librispeech --splits test-clean dev-clean
"""

import argparse
import json
import soundfile as sf
from pathlib import Path

import torchaudio


AVAILABLE_SPLITS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

DEFAULT_SPLITS = ["test-clean", "dev-clean"]

# LibriSpeech is always 16 kHz
SAMPLE_RATE = 16000


def download_split(root: Path, split: str):
    """Download a single LibriSpeech split."""
    print(f"Downloading LibriSpeech split: {split}")
    torchaudio.datasets.LIBRISPEECH(
        root=str(root),
        url=split,
        download=True,
    )
    print(f"  Download complete.")


def build_manifest(root: Path, split: str, output_path: Path):
    """Build a JSON-lines manifest by parsing the LibriSpeech directory and transcript files.

    Each line contains:
        - audio_path: path to the .flac file
        - speaker_id: LibriSpeech speaker ID
        - chapter_id: chapter ID
        - utterance_id: utterance string ID
        - transcript: normalized transcript text
        - num_frames: number of audio frames
        - sample_rate: audio sample rate
        - duration_s: duration in seconds
    """
    manifest_file = output_path / f"{split}.jsonl"
    print(f"Building manifest: {manifest_file}")

    # LibriSpeech directory: root/LibriSpeech/<split>/<speaker>/<chapter>/
    split_dir = root / "LibriSpeech" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    count = 0
    with open(manifest_file, "w", encoding="utf-8") as f:
        # Iterate over transcript files
        for trans_file in sorted(split_dir.rglob("*.trans.txt")):
            chapter_dir = trans_file.parent
            with open(trans_file, "r", encoding="utf-8") as tf:
                for line in tf:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: "<utt_id> <transcript>"
                    utt_id, transcript = line.split(" ", 1)
                    parts = utt_id.split("-")
                    speaker_id = int(parts[0])
                    chapter_id = int(parts[1])

                    audio_path = chapter_dir / f"{utt_id}.flac"
                    num_frames = sf.info(str(audio_path)).frames

                    entry = {
                        "audio_path": str(audio_path),
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utt_id,
                        "transcript": transcript.lower().strip(),
                        "num_frames": num_frames,
                        "sample_rate": SAMPLE_RATE,
                        "duration_s": round(num_frames / SAMPLE_RATE, 3),
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1

    print(f"  Wrote {count} entries to {manifest_file}")
    return manifest_file


def print_split_stats(manifest_path: Path):
    """Print basic statistics for a manifest file."""
    total_duration = 0.0
    speakers = set()
    count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            total_duration += entry["duration_s"]
            speakers.add(entry["speaker_id"])
            count += 1

    hours = total_duration / 3600
    print(f"  Stats: {count} utterances, {len(speakers)} speakers, {hours:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess LibriSpeech")
    parser.add_argument(
        "--root",
        type=str,
        default="./data/librispeech",
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
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {root.resolve()}")
    print(f"Splits: {args.splits}\n")

    for split in args.splits:
        download_split(root, split)
        manifest_path = build_manifest(root, split, manifest_dir)
        print_split_stats(manifest_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
