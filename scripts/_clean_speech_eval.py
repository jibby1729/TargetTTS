"""
Reproducible clean-speech (overlap=0.0) WER benchmark.

Tests three conditions on the same fixed clips:
  - clean:  raw target audio, no interferer
  - 0dB:    target + non-overlapping interferer at 0 dB SIR
  - 5dB:    target + non-overlapping interferer at 5 dB SIR

"Non-overlapping" means the interferer is placed in the audio after the target
speech ends (no temporal overlap). SIR is matched globally by RMS scaling.

On first run, PAIRS_CSV is written with fixed (test, enrollment, interferer)
triples deterministically selected from LibriSpeech test-clean. Every
subsequent run reloads that CSV — the same clips are always evaluated.

Per-condition results are saved to OUTPUT_CSV_TEMPLATE.
Pass --rebuild-pairs to regenerate the pairs CSV.
Pass --force to re-run inference even if results already exist.

Usage:
    uv run python scripts/_clean_speech_eval.py
    uv run python scripts/_clean_speech_eval.py --checkpoint checkpoints/curriculum_medium/best_model.pt
    uv run python scripts/_clean_speech_eval.py --rebuild-pairs --force
"""
import argparse
import json
import os
import random

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.extraction_net import ExtractionNet, apply_mask_and_istft, compute_stft
from src.preprocess import TARGET_SAMPLE_RATE, normalize_text, preprocess_audio
from src.speaker_encoder import SpeakerEncoder

SEED = 42
N_SAMPLES = 200
MANIFEST = "data/librispeech/manifests/test-clean.jsonl"
PAIRS_CSV = "data/eval_results/clean_speech_pairs.csv"
SIR_LEVELS = [None, 0, 5]   # None = clean (no interferer)


def build_fixed_pairs(manifest_path, n_samples, seed):
    """Deterministically select fixed (test, enrollment, interferer) triples."""
    with open(manifest_path) as f:
        records = [json.loads(line) for line in f]

    by_speaker = {}
    for r in records:
        by_speaker.setdefault(r["speaker_id"], []).append(r)

    # Sort utterances within each speaker for determinism, require >=2 utts
    eligible = {spk: sorted(utts, key=lambda u: u["utterance_id"])
                for spk, utts in by_speaker.items() if len(utts) >= 2}

    rng = random.Random(seed)
    speakers = sorted(eligible.keys())
    rng.shuffle(speakers)

    pairs = []
    for i, spk in enumerate(speakers):
        utts = eligible[spk][:]
        rng.shuffle(utts)
        # Pick an interferer from a different speaker (next in shuffled list)
        interferer_spk = speakers[(i + 1) % len(speakers)]
        interferer_utt = eligible[interferer_spk][0]

        pairs.append({
            "speaker_id": spk,
            "test_audio_path": utts[0]["audio_path"],
            "test_transcript": utts[0]["transcript"],
            "enrollment_audio_path": utts[1]["audio_path"],
            "interferer_audio_path": interferer_utt["audio_path"],
        })
        if len(pairs) >= n_samples:
            break

    df = pd.DataFrame(pairs)
    os.makedirs(os.path.dirname(PAIRS_CSV), exist_ok=True)
    df.to_csv(PAIRS_CSV, index=False)
    print(f"Saved {len(df)} fixed pairs to {PAIRS_CSV}")
    return df


def load_or_build_pairs(manifest_path, n_samples, seed, rebuild=False):
    if not rebuild and os.path.exists(PAIRS_CSV):
        df = pd.read_csv(PAIRS_CSV)
        # Backward compat: regenerate if interferer column is missing
        if "interferer_audio_path" not in df.columns:
            print("Pairs CSV missing interferer column — rebuilding...")
            return build_fixed_pairs(manifest_path, n_samples, seed)
        print(f"Loaded {len(df)} fixed pairs from {PAIRS_CSV}")
        return df
    print("Building fixed pairs...")
    return build_fixed_pairs(manifest_path, n_samples, seed)


def mix_with_interferer(target_wav, interferer_path, sir_db):
    """
    Create a non-overlapping mixture: [target | interferer] scaled to sir_db.

    Returns a 1-D float32 numpy array at TARGET_SAMPLE_RATE.
    """
    interferer_wav, sr = sf.read(interferer_path, dtype="float32")
    if sr != TARGET_SAMPLE_RATE:
        import torchaudio
        interferer_wav = torchaudio.functional.resample(
            torch.from_numpy(interferer_wav), sr, TARGET_SAMPLE_RATE
        ).numpy()
    if interferer_wav.ndim > 1:
        interferer_wav = interferer_wav.mean(axis=1)

    target_rms = np.sqrt(np.mean(target_wav ** 2)) + 1e-8
    interferer_rms = np.sqrt(np.mean(interferer_wav ** 2)) + 1e-8
    # Scale interferer so its RMS is target_rms * 10^(-sir/20)
    scale = (target_rms / interferer_rms) * (10 ** (-sir_db / 20))
    interferer_scaled = interferer_wav * scale

    # Concatenate: target first, then interferer (zero temporal overlap)
    return np.concatenate([target_wav, interferer_scaled])


def print_summary(results_by_condition):
    """Print a cross-condition comparison table."""
    print(f"\n{'Condition':<12}  {'Baseline':>10}  {'Extracted':>10}  {'Delta':>8}")
    print("-" * 48)
    for label, (wers_base, wers_ext) in results_by_condition.items():
        base_med = np.median(wers_base) * 100
        ext_med = np.median(wers_ext) * 100
        delta = ext_med - base_med
        sign = "+" if delta >= 0 else ""
        print(f"{label:<12}  {base_med:>9.1f}%  {ext_med:>9.1f}%  {sign}{delta:>6.1f}pp")
    print(f"\n  n={len(next(iter(results_by_condition.values()))[0])} samples  seed={SEED}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/curriculum_medium/best_model.pt")
    parser.add_argument("--manifest", type=str, default=MANIFEST)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output-dir", type=str, default="data/eval_results")
    parser.add_argument("--rebuild-pairs", action="store_true",
                        help="Regenerate pairs CSV even if it already exists")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if results already exist")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs_df = load_or_build_pairs(
        args.manifest, args.n_samples, SEED, rebuild=args.rebuild_pairs
    )

    # Derive output CSV name from checkpoint
    ckpt_tag = os.path.basename(os.path.dirname(args.checkpoint))
    output_csv = os.path.join(args.output_dir, f"clean_speech_wer_{ckpt_tag}.csv")

    if not args.force and os.path.exists(output_csv):
        print(f"Results already exist at {output_csv} — loading (use --force to re-run)")
        df = pd.read_csv(output_csv)
        results = {}
        for label in df["condition"].unique():
            sub = df[df["condition"] == label]
            results[label] = (sub["wer_baseline"].values, sub["wer_extracted"].values)
        print_summary(results)
        return

    # Load models
    print("\nLoading models...")
    encoder = SpeakerEncoder(device=device)

    model = ExtractionNet(input_power=0.3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded extraction model: epoch {ckpt.get('epoch','?')}, val_loss {ckpt.get('val_loss', float('nan')):.6f}")

    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).to(device)
    whisper_model.eval()
    forced_ids = whisper_proc.get_decoder_prompt_ids(language="english", task="transcribe")

    def transcribe(audio_np):
        inputs = whisper_proc(audio_np, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = whisper_model.generate(inputs["input_features"], forced_decoder_ids=forced_ids)
        return normalize_text(whisper_proc.batch_decode(ids, skip_special_tokens=True)[0])

    # Pre-compute speaker embeddings (enrollment clips are the same across conditions)
    print("Pre-computing speaker embeddings...")
    emb_cache = {}
    for path in tqdm(pairs_df["enrollment_audio_path"].unique(), desc="  ECAPA"):
        emb_cache[path] = encoder.encode_file(path).unsqueeze(0).to(device)

    # Run all conditions
    all_records = []
    condition_labels = {None: "clean", 0: "0dB SIR", 5: "5dB SIR"}

    for sir in SIR_LEVELS:
        label = condition_labels[sir]
        print(f"\n=== Condition: {label} ===")

        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"  {label}"):
            ref = normalize_text(str(row["test_transcript"]))
            if not ref:
                continue

            # Build input audio for this condition
            clean_wav = preprocess_audio(row["test_audio_path"]).squeeze().numpy()
            if sir is None:
                audio_np = clean_wav
            else:
                audio_np = mix_with_interferer(clean_wav, row["interferer_audio_path"], sir)

            # Baseline: audio → Whisper
            pred_base = transcribe(audio_np)
            wer_base = jiwer.wer(ref, pred_base)

            # Extraction: audio → extractor → Whisper
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            emb = emb_cache[row["enrollment_audio_path"]]
            mag, phase = compute_stft(waveform)
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                mask = model(mag.to(device), emb)
            extracted_np = apply_mask_and_istft(
                mag.to(device), phase.to(device), mask
            ).squeeze(0).cpu().numpy()
            pred_ext = transcribe(extracted_np)
            wer_ext = jiwer.wer(ref, pred_ext)

            all_records.append({
                "condition": label,
                "sir_db": sir if sir is not None else -1,
                "speaker_id": row["speaker_id"],
                "test_audio_path": row["test_audio_path"],
                "reference": ref,
                "pred_baseline": pred_base,
                "pred_extracted": pred_ext,
                "wer_baseline": wer_base,
                "wer_extracted": wer_ext,
                "wer_delta": wer_ext - wer_base,
            })

    results_df = pd.DataFrame(all_records)
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved per-sample results to {output_csv}")

    results = {}
    for label in condition_labels.values():
        sub = results_df[results_df["condition"] == label]
        results[label] = (sub["wer_baseline"].values, sub["wer_extracted"].values)
    print_summary(results)


if __name__ == "__main__":
    main()
