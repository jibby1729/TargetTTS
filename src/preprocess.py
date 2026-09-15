"""
Audio and text preprocessing utilities for the TargetTTS pipeline.

Audio:
    - Load audio from file
    - Resample to target sample rate (default 16 kHz)
    - Normalize amplitude (peak normalization)

Text:
    - normalize_text: Whisper EnglishTextNormalizer for English WER
    - normalize_akan_text: Akan-safe cleaner for Twi LM / WAXAL text
"""

from __future__ import annotations

import re
import unicodedata

TARGET_SAMPLE_RATE = 16000

_text_normalizer = None


def _english_normalizer():
    """Lazy-load Whisper's English normalizer (heavy import)."""
    global _text_normalizer
    if _text_normalizer is None:
        from whisper.normalizers import EnglishTextNormalizer

        _text_normalizer = EnglishTextNormalizer()
    return _text_normalizer


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """Load an audio file and return as a torch tensor.

    Args:
        path: Path to the audio file.
        target_sr: Expected sample rate (raises if mismatch).

    Returns:
        Mono waveform tensor of shape (1, num_samples).
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        raise ValueError(
            f"Expected sample rate {target_sr}, got {sr}. "
            f"Resample the audio to {target_sr} Hz before calling this function."
        )

    # Shape: (1, num_samples)
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform


def normalize_amplitude(waveform):
    """Peak-normalize a waveform to [-1, 1].

    Args:
        waveform: Audio tensor of shape (1, num_samples).

    Returns:
        Peak-normalized waveform.
    """
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def preprocess_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """Load, resample, and peak-normalize an audio file.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        Preprocessed mono waveform tensor of shape (1, num_samples).
    """
    waveform = load_audio(path, target_sr)
    waveform = normalize_amplitude(waveform)
    return waveform


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize a transcript for English WER evaluation.

    Applies Whisper's EnglishTextNormalizer which:
        - Lowercases
        - Removes punctuation
        - Expands contractions ("don't" -> "do not")
        - Removes filler words (uh, um, hmm, etc.)
        - Converts English number words to digits
        - Strips extra whitespace

    Do not use this for Twi/Akan LM text. Use normalize_akan_text instead.
    """
    return _english_normalizer()(text)


# Punctuation stripped for Akan LM text, including apostrophes.
# Does not match Akan letters such as ɛ/ɔ (they are letters, not punctuation).
_AKAN_PUNCTUATION_RE = re.compile(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`…—–-]")
_AKAN_WHITESPACE_RE = re.compile(r"\s+")


def normalize_akan_text(text: str) -> str:
    """Normalize an Akan/Twi transcript for LM training and Twi WER.

    Unlike normalize_text / EnglishTextNormalizer, this does not expand
    English contractions, drop English fillers, rewrite number words to
    digits, or strip combining marks via NFKD. WAXAL ASR writes numbers
    as Twi words (baako, mmienu, …), which are left unchanged.

    Steps:
        - Unicode NFC
        - lowercase
        - replace punctuation (including apostrophes) with spaces
        - collapse whitespace
        - keep Akan letters (ɛ, ɔ, etc.)
        - keep one dataset row as one line (does not split on '.')
    """
    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = _AKAN_PUNCTUATION_RE.sub(" ", value)
    value = _AKAN_WHITESPACE_RE.sub(" ", value)
    return value.strip()


# ---------------------------------------------------------------------------
# Build in-memory cache for WAXAL dataset
# ---------------------------------------------------------------------------

def build_waxal_cache(disk_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> dict:
    """
    Loads the WAXAL HuggingFace dataset from disk and builds an in-memory
    audio cache keyed by filename (matching the paths stored in the mix CSVs).

    Args:
        disk_path: Path to the saved HuggingFace dataset directory.
        target_sr:  Sample rate to decode audio at (default 16kHz).

    Returns:
        Dict mapping filename keys to peak-normalized float32 numpy arrays.
    """
    import numpy as np
    from datasets import load_from_disk, Audio as HFAudio
    from tqdm import tqdm

    print(f"Loading WAXAL dataset from {disk_path}...")
    ds = load_from_disk(disk_path)["test"]

    # Extract raw file paths before decoding
    ds_undecoded = ds.cast_column("audio", HFAudio(decode=False))
    paths = [
        a.get("path") if isinstance(a, dict) else None
        for a in ds_undecoded["audio"]
    ]
    ds = ds.add_column("audio_path", paths)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=target_sr))

    cache = {}
    for item in tqdm(ds, desc="Building cache"):
        path = item["audio_path"]
        key = path if path is not None else f"waxal_id_{item['id']}"
        array = item["audio"]["array"].astype(np.float32)

        # Peak normalize to match preprocess_audio behaviour
        peak = np.abs(array).max()
        if peak > 0:
            array = array / peak
        cache[key] = array

    print(f"Cached {len(cache)} audio arrays.")
    return cache
