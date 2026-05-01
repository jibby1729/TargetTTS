"""
Latency benchmark for the extraction pipeline.

Picks a fixed set of test samples (reproducible via seed), measures per-sample
latency for each pipeline stage, stores results to CSV, and prints a distribution
summary. ECAPA embeddings are pre-computed (cached) to reflect realistic use.

Usage:
    uv run python scripts/_latency_bench.py
    uv run python scripts/_latency_bench.py --n-samples 100 --checkpoint checkpoints/curriculum_easy/best_model.pt
"""
import argparse
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.extraction_net import ExtractionNet, apply_mask_and_istft, compute_stft
from src.preprocess import TARGET_SAMPLE_RATE, preprocess_audio
from src.speaker_encoder import SpeakerEncoder

SEED = 42
N_WARMUP = 5


def cuda_time(fn):
    """Run fn once, return wall-clock ms with GPU synchronisation."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, result


def print_distribution(values, label, unit="ms"):
    arr = np.array(values)
    p = np.percentile(arr, [5, 25, 50, 75, 95, 99])
    print(f"\n{label}")
    print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  "
          f"[p5={p[0]:.1f}  p25={p[1]:.1f}  p50={p[2]:.1f}  "
          f"p75={p[3]:.1f}  p95={p[4]:.1f}  p99={p[5]:.1f}]  {unit}")


def ascii_histogram(values, label, n_bins=10):
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = max(counts)
    bar_width = 30
    print(f"\n{label} (n={len(arr)})")
    for i, (c, e0, e1) in enumerate(zip(counts, edges[:-1], edges[1:])):
        bar = "█" * int(bar_width * c / max_count) if max_count > 0 else ""
        print(f"  [{e0:6.1f}–{e1:6.1f}ms]  {bar} {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/curriculum_easy/best_model.pt")
    parser.add_argument("--test-csv", type=str, default="data/prepared_training/test.csv")
    parser.add_argument("--mix-dir", type=str, default="data/synthetic_mixtures/librispeech_extraction")
    parser.add_argument("--output-csv", type=str, default="data/eval_results/latency_benchmark.csv")
    args = parser.parse_args()

    device = "cuda"
    rng = np.random.default_rng(SEED)

    # --- Fixed sample selection ---
    df_test = pd.read_csv(args.test_csv)
    # Stratify by overlap ratio so we cover the full distribution
    groups = df_test.groupby("overlap_ratio")
    per_group = max(1, args.n_samples // len(groups))
    samples = pd.concat([
        g.sample(n=min(per_group, len(g)), random_state=SEED)
        for _, g in groups
    ]).reset_index(drop=True)
    print(f"Selected {len(samples)} fixed samples across overlap ratios:")
    print(samples["overlap_ratio"].value_counts().sort_index().to_string())

    # --- Load models ---
    print("\nLoading models...")
    encoder = SpeakerEncoder(device=device)

    model = ExtractionNet(input_power=0.3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded extraction model: epoch {ckpt.get('epoch', '?')}")

    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).to(device)
    whisper_model.eval()
    forced_ids = whisper_proc.get_decoder_prompt_ids(language="english", task="transcribe")

    # --- Pre-compute all speaker embeddings (cached scenario) ---
    print("Pre-computing speaker embeddings (cached scenario)...")
    unique_enrollments = samples["enrollment_audio_path"].unique()
    emb_cache = {}
    for path in tqdm(unique_enrollments, desc="  ECAPA"):
        emb_cache[path] = encoder.encode_file(path).unsqueeze(0).to(device)

    # --- GPU warmup with a dummy sample ---
    print("Warming up GPU...")
    _row = samples.iloc[0]
    _wav = preprocess_audio(f"{args.mix_dir}/{_row['mix_id']}.wav")
    _mag, _ph = compute_stft(_wav)
    _emb = emb_cache[_row["enrollment_audio_path"]]
    for _ in range(N_WARMUP):
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            _mask = model(_mag.to(device), _emb)
            apply_mask_and_istft(_mag.to(device), _ph.to(device), _mask)
        _inp = whisper_proc(_wav.squeeze().numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            whisper_model.generate(_inp["input_features"].to(device), forced_decoder_ids=forced_ids)
    torch.cuda.synchronize()

    # --- Per-sample timing ---
    print(f"\nBenchmarking {len(samples)} samples...")
    records = []

    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        mix_path = f"{args.mix_dir}/{row['mix_id']}.wav"
        audio_np, _ = sf.read(mix_path, dtype="float32")
        duration_s = len(audio_np) / TARGET_SAMPLE_RATE
        emb = emb_cache[row["enrollment_audio_path"]]

        # Load waveform
        waveform = preprocess_audio(mix_path)

        # STFT
        t_stft, (magnitude, phase) = cuda_time(lambda: compute_stft(waveform))
        mag_gpu = magnitude.to(device)
        phase_gpu = phase.to(device)

        # Extraction net forward + iSTFT
        with torch.no_grad():
            t_extract, extracted = cuda_time(lambda: (
                apply_mask_and_istft(
                    mag_gpu, phase_gpu,
                    model(mag_gpu, emb) if True else None  # bfloat16 via autocast below
                )
            ))
        # Redo cleanly with autocast
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            t_extract, _ = cuda_time(lambda: apply_mask_and_istft(
                mag_gpu, phase_gpu, model(mag_gpu, emb)
            ))
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            mask = model(mag_gpu, emb)
        extracted_np = apply_mask_and_istft(mag_gpu, phase_gpu, mask).squeeze(0).cpu().numpy()

        # Whisper on raw mixture
        inp_raw = whisper_proc(audio_np, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        inp_raw = {k: v.to(device) for k, v in inp_raw.items()}
        with torch.no_grad():
            t_whisper_raw, _ = cuda_time(
                lambda: whisper_model.generate(inp_raw["input_features"], forced_decoder_ids=forced_ids)
            )

        # Whisper on extracted audio
        inp_ext = whisper_proc(extracted_np, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        inp_ext = {k: v.to(device) for k, v in inp_ext.items()}
        with torch.no_grad():
            t_whisper_ext, _ = cuda_time(
                lambda: whisper_model.generate(inp_ext["input_features"], forced_decoder_ids=forced_ids)
            )

        records.append({
            "mix_id": row["mix_id"],
            "overlap_ratio": row["overlap_ratio"],
            "sir_level_db": row["sir_level_db"],
            "duration_s": duration_s,
            "t_stft_ms": t_stft,
            "t_extract_ms": t_extract,
            "t_overhead_ms": t_stft + t_extract,   # extra cost vs baseline
            "t_whisper_raw_ms": t_whisper_raw,
            "t_whisper_ext_ms": t_whisper_ext,
            "rtf_overhead": (t_stft + t_extract) / (duration_s * 1000),
        })

    # --- Save and report ---
    df_results = pd.DataFrame(records)
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nSaved per-sample results to {args.output_csv}")

    print("\n" + "=" * 60)
    print("LATENCY DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"\nAudio duration range: {df_results['duration_s'].min():.1f}s – {df_results['duration_s'].max():.1f}s")

    print_distribution(df_results["t_stft_ms"], "STFT (mix → spectrogram)")
    print_distribution(df_results["t_extract_ms"], "Extraction net + iSTFT")
    print_distribution(df_results["t_overhead_ms"], "Total overhead (STFT + net + iSTFT)")
    print_distribution(df_results["t_whisper_raw_ms"], "Whisper on raw mixture")
    print_distribution(df_results["t_whisper_ext_ms"], "Whisper on extracted audio")
    print_distribution(df_results["rtf_overhead"] * 100, "Overhead RTF (%)", unit="%")

    ascii_histogram(df_results["t_overhead_ms"], "Extraction overhead distribution")
    ascii_histogram(df_results["t_whisper_raw_ms"], "Whisper latency distribution (raw)")

    print("\nPer overlap ratio — median overhead:")
    print(df_results.groupby("overlap_ratio")[["t_overhead_ms", "t_whisper_raw_ms"]].median().to_string())


if __name__ == "__main__":
    main()
