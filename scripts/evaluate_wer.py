"""
Evaluate WER with and without speaker extraction.

Runs the full pipeline on the test split:
  - Baseline: raw mixture → Whisper → WER
  - Extraction: mixture → extraction network → Whisper → WER

Produces per-sample CSVs, aggregated stats, and comparison plots.

Usage:
    python -m scripts.evaluate_wer --checkpoint checkpoints/best_model.pt
    python -m scripts.evaluate_wer --checkpoint checkpoints/best_model.pt --overlap-ratios 0.5 --sir-levels 0
"""

import argparse
import os

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.extraction_net import ExtractionNet, compute_stft, apply_mask_and_istft
from src.preprocess import preprocess_audio, normalize_text, TARGET_SAMPLE_RATE
from src.speaker_encoder import SpeakerEncoder


# Plot styling (matches scripts/plot_wer.py)
SIR_COLORS = {0: "#E53935", 5: "#1E88E5", 10: "#43A047"}
SIR_MARKERS = {0: "s", 5: "o", 10: "^"}


def load_extraction_model(checkpoint_path, input_power, device):
    """Load trained ExtractionNet from checkpoint."""
    model = ExtractionNet(input_power=input_power).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded extraction model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def load_whisper(model_id, device):
    """Load Whisper model and processor from HuggingFace transformers."""
    is_local = os.path.isdir(model_id)
    processor = WhisperProcessor.from_pretrained(model_id, local_files_only=is_local)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, local_files_only=is_local
    ).to(device)
    model.eval()
    print(f"Loaded Whisper: {model_id}")
    return processor, model


def get_or_compute_embeddings(test_df, cache_path, device):
    """Load cached embeddings or compute via SpeakerEncoder."""
    unique_paths = test_df["enrollment_audio_path"].unique()

    if os.path.exists(cache_path):
        embeddings = torch.load(cache_path, weights_only=False)
        missing = [p for p in unique_paths if p not in embeddings]
    else:
        embeddings = {}
        missing = list(unique_paths)

    if missing:
        print(f"Computing {len(missing)} missing speaker embeddings...")
        encoder = SpeakerEncoder(device=device)
        for path in tqdm(missing, desc="  embeddings"):
            embeddings[path] = encoder.encode_file(path)
        torch.save(embeddings, cache_path)
        print(f"Cached embeddings to {cache_path}")
    else:
        print(f"Loaded {len(embeddings)} cached embeddings from {cache_path}")

    return embeddings


def calculate_leakage(pred_text, target_text, noise_text):
    """Percentage of noise-specific words that leaked into the prediction."""
    pred_words = set(pred_text.split())
    target_words = set(target_text.split())
    noise_words = set(noise_text.split())

    noise_only = noise_words - target_words
    if len(noise_only) == 0:
        return 0.0

    leaked = pred_words & noise_only
    return len(leaked) / len(noise_only)


@torch.no_grad()
def extract_audio(mix_path, enrollment_path, model, embeddings, device):
    """Run extraction pipeline: load → STFT → mask → iSTFT → numpy array."""
    waveform = preprocess_audio(mix_path)  # (1, N)
    magnitude, phase = compute_stft(waveform)  # (1, 257, T)

    emb = embeddings[enrollment_path].unsqueeze(0).to(device)  # (1, 192)
    mask = model(magnitude.to(device), emb)  # (1, 257, T)

    extracted = apply_mask_and_istft(
        magnitude.to(device), phase.to(device), mask
    )  # (1, N')
    return extracted.squeeze(0).cpu().numpy()


def transcribe(audio_array, processor, model, language, device):
    """Transcribe audio array using HF Whisper."""
    inputs = processor(
        audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    lang_name = {"en": "english", "twi": "english"}.get(language, "english")
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang_name, task="transcribe"
    )

    predicted_ids = model.generate(
        inputs["input_features"], forced_decoder_ids=forced_decoder_ids
    )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return normalize_text(text)


def run_evaluation(test_df, mix_dir, extraction_model, embeddings,
                   whisper_proc, whisper_model, language, device,
                   output_csv, resume):
    """Evaluate all test samples, with checkpointing."""
    completed_ids = set()
    results = []

    if resume and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        completed_ids = set(existing["mix_id"])
        results = existing.to_dict("records")
        print(f"Resuming: {len(completed_ids)} already completed")

    pending = test_df[~test_df["mix_id"].isin(completed_ids)]
    mode = "extraction" if extraction_model is not None else "baseline"

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending),
                                       desc=f"  {mode}")):
        mix_id = row["mix_id"]
        mix_path = os.path.join(mix_dir, f"{mix_id}.wav")
        target_text = normalize_text(str(row["target_transcript"]))
        noise_text = normalize_text(str(row["noise_transcript"]))

        if extraction_model is not None:
            audio = extract_audio(
                mix_path, row["enrollment_audio_path"],
                extraction_model, embeddings, device
            )
        else:
            audio, _ = sf.read(mix_path, dtype="float32")

        pred_text = transcribe(audio, whisper_proc, whisper_model, language, device)

        wer_val = jiwer.wer(target_text, pred_text) if target_text else 1.0
        cer_val = jiwer.cer(target_text, pred_text) if target_text else 1.0
        leakage = calculate_leakage(pred_text, target_text, noise_text)

        results.append({
            "mix_id": mix_id,
            "sir_level_db": row["sir_level_db"],
            "overlap_ratio": row["overlap_ratio"],
            "target_transcript": target_text,
            "pred_transcript": pred_text,
            "wer": wer_val,
            "cer": cer_val,
            "leakage_rate": leakage,
        })

        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


def aggregate_results(df):
    """Group by (sir_level_db, overlap_ratio) and compute mean/median/std metrics."""
    grouped = df.groupby(["sir_level_db", "overlap_ratio"])
    agg = grouped[["wer", "cer", "leakage_rate"]].mean().reset_index()
    medians = grouped[["wer", "cer"]].median().reset_index()
    stds = grouped[["wer", "cer"]].std().reset_index()
    agg["wer_median"] = medians["wer"]
    agg["cer_median"] = medians["cer"]
    agg["wer_std"] = stds["wer"]
    agg["cer_std"] = stds["cer"]
    agg["n_samples"] = grouped.size().values
    return agg


def plot_comparison(extraction_agg, baseline_agg, output_dir):
    """Generate WER comparison plot: extraction (solid) vs baseline (dashed)."""
    sir_levels = sorted(set(extraction_agg["sir_level_db"]) | set(baseline_agg["sir_level_db"]))

    plt.figure(figsize=(8, 5))

    for sir in sir_levels:
        color = SIR_COLORS.get(sir, "gray")
        marker = SIR_MARKERS.get(sir, "o")

        base = baseline_agg[baseline_agg["sir_level_db"] == sir].sort_values("overlap_ratio")
        ext = extraction_agg[extraction_agg["sir_level_db"] == sir].sort_values("overlap_ratio")

        if not base.empty:
            plt.plot(
                base["overlap_ratio"], base["wer"] * 100,
                marker=marker, color=color, linewidth=2, markersize=6,
                linestyle="--", alpha=0.6,
                label=f"Baseline SIR={'+' if sir >= 0 else ''}{sir}dB",
            )
        if not ext.empty:
            plt.plot(
                ext["overlap_ratio"], ext["wer"] * 100,
                marker=marker, color=color, linewidth=2, markersize=6,
                linestyle="-",
                label=f"Extracted SIR={'+' if sir >= 0 else ''}{sir}dB",
            )

    plt.xlabel("Overlap Ratio", fontsize=12)
    plt.ylabel("WER (%)", fontsize=12)
    plt.title("WER: Extraction vs Baseline", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "wer_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary(extraction_agg, baseline_agg):
    """Print a side-by-side comparison table with mean, median, and std."""
    merged = extraction_agg.merge(
        baseline_agg, on=["sir_level_db", "overlap_ratio"],
        suffixes=("_ext", "_base"), how="outer"
    ).sort_values(["sir_level_db", "overlap_ratio"])

    print(f"\n{'SIR':>5}  {'Overlap':>8}  {'n':>5}  {'Base Mean':>18}  {'Base Median':>13}  {'Ext Mean':>18}  {'Ext Median':>13}  {'Delta(Med)':>11}")
    print("-" * 105)
    for _, r in merged.iterrows():
        n = int(r.get("n_samples_ext", r.get("n_samples_base", 0)))
        base_mean = r.get("wer_base", float("nan")) * 100
        base_med = r.get("wer_median_base", float("nan")) * 100
        base_std = r.get("wer_std_base", float("nan")) * 100
        ext_mean = r.get("wer_ext", float("nan")) * 100
        ext_med = r.get("wer_median_ext", float("nan")) * 100
        ext_std = r.get("wer_std_ext", float("nan")) * 100
        delta = ext_med - base_med
        sign = "+" if delta >= 0 else ""
        print(f"{int(r['sir_level_db']):>5}  {r['overlap_ratio']:>8.1f}  {n:>5}  "
              f"{base_mean:>6.1f}% ±{base_std:>5.1f}%  {base_med:>11.1f}%  "
              f"{ext_mean:>6.1f}% ±{ext_std:>5.1f}%  {ext_med:>11.1f}%  "
              f"{sign}{delta:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER with/without extraction")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to extraction model checkpoint (omit for baseline-only)")
    parser.add_argument("--test-csv", type=str, default="data/prepared_training/test.csv")
    parser.add_argument("--mix-dir", type=str,
                        default="data/synthetic_mixtures/librispeech_extraction")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-tiny")
    parser.add_argument("--language", type=str, default="en", choices=["en", "twi"])
    parser.add_argument("--overlap-ratios", type=float, nargs="*", default=None)
    parser.add_argument("--sir-levels", type=float, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="data/eval_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    test_df = pd.read_csv(args.test_csv)
    if args.overlap_ratios:
        test_df = test_df[test_df["overlap_ratio"].isin(args.overlap_ratios)]
    if args.sir_levels:
        test_df = test_df[test_df["sir_level_db"].isin(args.sir_levels)]
    print(f"Test samples: {len(test_df)}")
    print(f"  SIR levels: {sorted(test_df['sir_level_db'].unique())}")
    print(f"  Overlap ratios: {sorted(test_df['overlap_ratio'].unique())}")
    print()

    # Load models
    extraction_model = None
    if args.checkpoint:
        extraction_model = load_extraction_model(args.checkpoint, input_power=0.3, device=device)

    whisper_proc, whisper_model = load_whisper(args.whisper_model, device)

    # Get/compute speaker embeddings (needed for extraction; compute anyway for caching)
    cache_path = os.path.join(os.path.dirname(args.test_csv), "test_embeddings.pt")
    embeddings = get_or_compute_embeddings(test_df, cache_path, device) if extraction_model else {}
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run baseline (no extraction)
    print("=== Baseline (no extraction) ===")
    baseline_csv = os.path.join(args.output_dir, "baseline_per_sample.csv")
    baseline_df = run_evaluation(
        test_df, args.mix_dir, None, embeddings,
        whisper_proc, whisper_model, args.language, device,
        baseline_csv, args.resume,
    )
    baseline_agg = aggregate_results(baseline_df)
    baseline_agg.to_csv(os.path.join(args.output_dir, "baseline_aggregated.csv"), index=False)

    if extraction_model is None:
        print("\nNo checkpoint provided — baseline-only mode.")
        print(f"Results saved to {args.output_dir}/")
        return

    # Run extraction
    print("\n=== With extraction ===")
    extraction_csv = os.path.join(args.output_dir, "extraction_per_sample.csv")
    extraction_df = run_evaluation(
        test_df, args.mix_dir, extraction_model, embeddings,
        whisper_proc, whisper_model, args.language, device,
        extraction_csv, args.resume,
    )
    extraction_agg = aggregate_results(extraction_df)
    extraction_agg.to_csv(os.path.join(args.output_dir, "extraction_aggregated.csv"), index=False)

    # Summary and plot
    print_summary(extraction_agg, baseline_agg)
    plot_comparison(extraction_agg, baseline_agg, args.output_dir)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
