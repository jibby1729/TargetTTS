"""
Audio and text preprocessing utilities for the TargetTTS pipeline.

Audio:
    - Load audio from file
    - Resample to target sample rate (default 16 kHz)
    - Normalize amplitude (peak normalization)

Text:
    - Normalize transcripts using Whisper's EnglishTextNormalizer
      (lowercase, strip punctuation, expand contractions, remove fillers)
    - This ensures fair WER comparison between ground truth and ASR output
"""

import numpy as np
import soundfile as sf
import torch
from whisper.normalizers import EnglishTextNormalizer

TARGET_SAMPLE_RATE = 16000

# Singleton normalizer instance
_text_normalizer = EnglishTextNormalizer()


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file and return as a torch tensor.

    Args:
        path: Path to the audio file.
        target_sr: Expected sample rate (raises if mismatch).

    Returns:
        Mono waveform tensor of shape (1, num_samples).
    """
    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        raise ValueError(
            f"Expected sample rate {target_sr}, got {sr}. "
            f"Resampling not yet implemented — LibriSpeech should always be 16 kHz."
        )

    # Shape: (1, num_samples)
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform


def normalize_amplitude(waveform: torch.Tensor) -> torch.Tensor:
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


def preprocess_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
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
    """Normalize a transcript for WER evaluation.

    Applies Whisper's EnglishTextNormalizer which:
        - Lowercases
        - Removes punctuation
        - Expands contractions ("don't" -> "do not")
        - Removes filler words (uh, um, hmm, etc.)
        - Strips extra whitespace

    Args:
        text: Raw transcript string.

    Returns:
        Normalized transcript string.
    """
    return _text_normalizer(text)
