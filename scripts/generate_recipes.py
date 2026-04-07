"""
Generate mixture recipes with enrollment utterances for extraction training.

For each mix, selects:
  - A target utterance
  - An interferer utterance from a different speaker
  - An enrollment utterance from the same speaker as the target (but different utterance)

Usage:
    python -m scripts.generate_recipes
    python -m scripts.generate_recipes --num-mixes 5000 --sir-levels 0 5 10
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_SIR_LEVELS = [0, 5]
DEFAULT_OVERLAP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
DEFAULT_NUM_MIXES = 12500


def load_librispeech_metadata(manifest_path):
    """Load audio pool entries from a LibriSpeech JSONL manifest."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            entries.append({
                "dataset": "librispeech",
                "filepath": data["audio_path"],
                "speaker_id": str(data["speaker_id"]),
                "transcript": data["transcript"],
                "duration_s": data["duration_s"],
            })
    print(f"Loaded {len(entries)} LibriSpeech entries.")
    return entries


def build_speaker_index(entries):
    """Group entries by speaker_id for enrollment selection."""
    index = defaultdict(list)
    for e in entries:
        index[e["speaker_id"]].append(e)
    return index


def generate_mix_recipes(audio_pool, num_mixes, sir_levels, overlap_ratios):
    """
    Randomly pair target and interferer utterances from different speakers,
    assigning random SIR levels and overlap ratios.
    Also selects an enrollment utterance from the same speaker as the target.
    """
    speaker_index = build_speaker_index(audio_pool)

    # Only use speakers that have at least 2 utterances (need one for target, one for enrollment)
    valid_speakers = [sid for sid, utts in speaker_index.items() if len(utts) >= 2]
    valid_pool = [e for e in audio_pool if e["speaker_id"] in valid_speakers]

    if len(valid_speakers) < 2:
        raise ValueError("Need at least 2 speakers with 2+ utterances each.")

    print(f"  {len(valid_speakers)} speakers with 2+ utterances, {len(valid_pool)} total utterances")

    recipes = []
    for i in range(num_mixes):
        # Pick target utterance
        target = random.choice(valid_pool)

        # Pick enrollment: different utterance, same speaker
        same_speaker_utts = speaker_index[target["speaker_id"]]
        enrollment = random.choice(same_speaker_utts)
        while enrollment["filepath"] == target["filepath"]:
            enrollment = random.choice(same_speaker_utts)

        # Pick interferer: different speaker
        interferer = random.choice(audio_pool)
        while interferer["speaker_id"] == target["speaker_id"]:
            interferer = random.choice(audio_pool)

        recipes.append({
            "mix_id": f"mix_{target['dataset']}_{i:05d}",
            "dataset": target["dataset"],
            "target_audio_path": target["filepath"],
            "target_speaker_id": target["speaker_id"],
            "target_transcript": target["transcript"],
            "enrollment_audio_path": enrollment["filepath"],
            "noise_audio_path": interferer["filepath"],
            "noise_speaker_id": interferer["speaker_id"],
            "noise_transcript": interferer["transcript"],
            "sir_level_db": random.choice(sir_levels),
            "overlap_ratio": random.choice(overlap_ratios),
        })

    return pd.DataFrame(recipes)


def main():
    parser = argparse.ArgumentParser(description="Generate mixture recipes with enrollment utterances")
    parser.add_argument("--manifest", type=str, default="data/librispeech/manifests/test-clean.jsonl",
                        help="Path to LibriSpeech JSONL manifest")
    parser.add_argument("--output", type=str, default="data/mixture_recipes/librispeech_extraction_recipes.csv",
                        help="Output CSV path")
    parser.add_argument("--num-mixes", type=int, default=DEFAULT_NUM_MIXES)
    parser.add_argument("--sir-levels", type=int, nargs="+", default=DEFAULT_SIR_LEVELS)
    parser.add_argument("--overlap-ratios", type=float, nargs="+", default=DEFAULT_OVERLAP_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"SIR levels: {args.sir_levels}")
    print(f"Overlap ratios: {args.overlap_ratios}")
    print(f"Num mixes: {args.num_mixes}")
    print()

    entries = load_librispeech_metadata(args.manifest)
    df = generate_mix_recipes(entries, args.num_mixes, args.sir_levels, args.overlap_ratios)

    # Print distribution
    dist = pd.crosstab(df["sir_level_db"], df["overlap_ratio"])
    print(f"\nMix distribution:\n{dist.to_string()}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} recipes to {args.output}")


if __name__ == "__main__":
    main()
