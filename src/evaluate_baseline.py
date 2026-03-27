"""
Evaluate pretrained Whisper models on LibriSpeech clean splits.

Runs Whisper inference on each utterance in a manifest, computes WER
using jiwer, and prints per-model results.

Usage:
    python src/evaluate_baseline.py [--splits SPLIT ...] [--models MODEL ...]

Examples:
    python src/evaluate_baseline.py
    python src/evaluate_baseline.py --splits test-clean --models tiny
    python src/evaluate_baseline.py --splits test-clean dev-clean --models tiny base
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import whisper
from jiwer import wer

from src.preprocess import preprocess_audio, normalize_text, TARGET_SAMPLE_RATE


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(model_name: str, manifest_path: Path, device: str) -> dict:
    """Run Whisper on all utterances in a manifest and return WER stats."""
    entries = load_manifest(manifest_path)
    print(f"\n  Loading whisper-{model_name} on {device}...", flush=True)
    model = whisper.load_model(model_name, device=device)

    references = []
    hypotheses = []

    start = time.time()
    for i, entry in enumerate(entries):
        audio_path = entry["audio_path"]
        ref_text = normalize_text(entry["transcript"])

        # Load audio with our preprocessing (avoids ffmpeg dependency)
        waveform = preprocess_audio(audio_path)
        audio_np = waveform.squeeze(0).numpy()  # (num_samples,)
        result = model.transcribe(audio_np, language="en")
        hyp_text = normalize_text(result["text"])

        references.append(ref_text)
        hypotheses.append(hyp_text)

        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - start
            running_wer = wer(references, hypotheses)
            print(f"    [{i+1}/{len(entries)}] WER so far: {running_wer*100:.2f}% | {elapsed:.1f}s elapsed", flush=True)

    elapsed = time.time() - start
    total_wer = wer(references, hypotheses)

    return {
        "model": model_name,
        "split": manifest_path.stem,
        "num_utterances": len(entries),
        "wer": total_wer,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper baselines on LibriSpeech")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test-clean"],
        help="Manifest splits to evaluate (default: test-clean)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "base"],
        help="Whisper model sizes to evaluate (default: tiny base)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/librispeech/manifests",
        help="Directory containing JSONL manifests",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(args.manifest_dir)

    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Splits: {args.splits}")

    results = []
    for split in args.splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            print(f"\nWARNING: Manifest not found: {manifest_path}, skipping.")
            continue

        for model_name in args.models:
            print(f"\n{'='*60}")
            print(f"Evaluating whisper-{model_name} on {split}")
            print(f"{'='*60}")

            result = evaluate(model_name, manifest_path, device)
            results.append(result)

            print(f"\n  WER: {result['wer']:.4f} ({result['wer']*100:.2f}%)")
            print(f"  Utterances: {result['num_utterances']}")
            print(f"  Time: {result['elapsed_s']}s")

    # Summary table
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Split':<14} {'WER':>8} {'Utterances':>12} {'Time':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['model']:<12} {r['split']:<14} {r['wer']*100:>7.2f}% {r['num_utterances']:>12} {r['elapsed_s']:>7.1f}s")

    # Save results to JSON
    output_path = Path("results")
    output_path.mkdir(exist_ok=True)
    results_file = output_path / "baseline_wer.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
