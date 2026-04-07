"""
Generate synthetic mixtures with aligned clean sources for extraction training.

For each recipe row, saves:
  - {mix_id}.wav          — the mixture (target + scaled interferer)
  - {mix_id}_target.wav   — aligned clean target (same timeline as mixture)
  - {mix_id}_interf.wav   — aligned clean interferer (same timeline as mixture)

Enrollment audio is not copied — the recipe CSV points to the original file.

Uses the same mixing logic as ADL_Generate_Mixtures.ipynb:
  - Active RMS-based SIR scaling
  - 2-4 random noise snippets with Hann fades
  - Peak normalization to prevent clipping

Usage:
    python -m scripts.generate_mixtures
    python -m scripts.generate_mixtures --csv data/mixture_recipes/librispeech_extraction_recipes.csv
"""

import argparse
import os
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.preprocess import preprocess_audio, TARGET_SAMPLE_RATE

TARGET_SR = TARGET_SAMPLE_RATE
FADE_DURATION_MS = 10
FADE_SAMPLES = int((FADE_DURATION_MS / 1000) * TARGET_SR)


def get_active_rms(y, top_db=30):
    """Calculates RMS energy only on the non-silent parts of the audio."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return np.sqrt(np.mean(y**2)) + 1e-8
    active = np.concatenate([y[s:e] for s, e in intervals])
    rms = np.sqrt(np.mean(active**2))
    return rms + 1e-8


def apply_fade(snippet):
    """Applies a Hann window fade-in and fade-out to prevent audio clicks."""
    if len(snippet) < FADE_SAMPLES * 2:
        return snippet
    window = np.hanning(FADE_SAMPLES * 2)
    snippet_faded = snippet.copy()
    snippet_faded[:FADE_SAMPLES] *= window[:FADE_SAMPLES]
    snippet_faded[-FADE_SAMPLES:] *= window[FADE_SAMPLES:]
    return snippet_faded


def apply_energy_scaling(y_t, y_n, sir_db):
    """Rescales the noise waveform to achieve the target SIR."""
    rms_t = get_active_rms(y_t)
    rms_n = get_active_rms(y_n)
    scale = (rms_t / (10 ** (sir_db / 20))) / rms_n
    return y_n * scale


def mix_audio_aligned(y_t, y_n, sir_db, overlap_ratio):
    """
    Mix target and interferer, returning the mixture and both aligned clean sources.

    Returns:
        y_mix: the mixture waveform (same length as y_t)
        y_target_aligned: clean target on the mixture timeline (same as y_t)
        y_interf_aligned: clean interferer on the mixture timeline (zeros where no interference)
        peak_scale: the peak normalization factor applied (so aligned sources match)
    """
    if overlap_ratio == 0.0:
        return y_t.copy(), y_t.copy(), np.zeros_like(y_t), 1.0

    # Scale interferer to target SIR
    y_n_scaled = apply_energy_scaling(y_t, y_n, sir_db)

    len_t = len(y_t)
    len_n = len(y_n_scaled)
    target_overlap_samples = int(len_t * overlap_ratio)

    # Break noise into 2-4 random snippets
    num_snippets = random.randint(2, 4)
    snippet_length = target_overlap_samples // num_snippets
    chunk_size = len_t // num_snippets

    y_mix = y_t.copy()
    y_interf_aligned = np.zeros(len_t, dtype=np.float32)
    noise_pointer = 0

    for i in range(num_snippets):
        if noise_pointer + snippet_length > len_n:
            noise_pointer = 0

        snippet = y_n_scaled[noise_pointer:noise_pointer + snippet_length]
        noise_pointer += snippet_length

        snippet = apply_fade(snippet)

        max_start = chunk_size - len(snippet)
        start_pos = (i * chunk_size) + (random.randint(0, max_start) if max_start > 0 else 0)
        end_pos = start_pos + len(snippet)

        if end_pos <= len_t:
            y_mix[start_pos:end_pos] += snippet
            y_interf_aligned[start_pos:end_pos] = snippet
        else:
            fit_length = len_t - start_pos
            y_mix[start_pos:] += snippet[:fit_length]
            y_interf_aligned[start_pos:] = snippet[:fit_length]

    # Peak normalization — apply same scale to all signals for consistency
    peak_scale = 1.0
    max_val = np.max(np.abs(y_mix))
    if max_val >= 1.0:
        peak_scale = 0.9 / max_val
        y_mix = y_mix * peak_scale

    # Apply same scaling to aligned sources so they sum to the mixture
    y_target_aligned = y_t * peak_scale
    y_interf_aligned = y_interf_aligned * peak_scale

    return y_mix, y_target_aligned, y_interf_aligned, peak_scale


def load_audio_np(path):
    """Load audio file as numpy array using our preprocessing."""
    waveform = preprocess_audio(path)
    return waveform.squeeze(0).numpy()


def run_generation(csv_path, output_dir, base_path="."):
    """Generate mixtures with aligned sources for all rows in the recipe CSV."""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    skipped, errors = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        mix_id = row["mix_id"]
        sir = row["sir_level_db"]
        overlap = row["overlap_ratio"]

        mix_path = os.path.join(output_dir, f"{mix_id}.wav")
        target_path = os.path.join(output_dir, f"{mix_id}_target.wav")
        interf_path = os.path.join(output_dir, f"{mix_id}_interf.wav")

        # Skip if all 3 files exist
        if os.path.exists(mix_path) and os.path.exists(target_path) and os.path.exists(interf_path):
            skipped += 1
            continue

        try:
            # Resolve paths
            t_path = row["target_audio_path"]
            n_path = row["noise_audio_path"]
            if not os.path.isabs(t_path):
                t_path = os.path.join(base_path, t_path)
            if not os.path.isabs(n_path):
                n_path = os.path.join(base_path, n_path)

            y_t = load_audio_np(t_path)
            y_n = load_audio_np(n_path)

            y_mix, y_target_aligned, y_interf_aligned, _ = mix_audio_aligned(
                y_t, y_n, sir, overlap
            )

            sf.write(mix_path, y_mix, TARGET_SR)
            sf.write(target_path, y_target_aligned, TARGET_SR)
            sf.write(interf_path, y_interf_aligned, TARGET_SR)

        except Exception as e:
            print(f"Error processing {mix_id}: {e}")
            errors += 1

    print(f"\nDone. Generated: {len(df) - skipped - errors} | Skipped: {skipped} | Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mixtures with aligned sources")
    parser.add_argument("--csv", type=str, default="data/mixture_recipes/librispeech_extraction_recipes.csv",
                        help="Recipe CSV path")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_mixtures/librispeech_extraction",
                        help="Output directory for wav files")
    parser.add_argument("--base-path", type=str, default=".",
                        help="Base path to resolve relative audio paths")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Recipe CSV: {args.csv}")
    print(f"Output dir: {args.output_dir}")
    print()

    run_generation(args.csv, args.output_dir, args.base_path)


if __name__ == "__main__":
    main()
