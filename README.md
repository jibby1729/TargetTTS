# TargetTTS: SNR Branch

ECE1508 Winter course project on isolating a target speaker's transcript from overlapping speech, benchmarked across a range of Signal-to-Interference Ratios (SIR) and overlap conditions.

This branch adds:
- WAXAL (Twi) dataset download, preprocessing, and in-memory cache utilities
- Whisper-small model finetuned for Twi ASR on WAXAL dataset
- Synthetic mixture generation for both LibriSpeech (English) and WAXAL (Twi)
- Mixture benchmarking pipeline evaluating Whisper small on both datasets
- Shared preprocessing and text normalization via `src/preprocess.py`

---

## Project Structure


```
TargetTTS/
├── data/
│   ├── download_librispeech.py       # Download LibriSpeech + build JSONL manifests
│   └── download_waxal.py             # Download WAXAL dataset from HuggingFace Hub
│   └── mixture_recipes/              # Generated mix recipe CSVs
│   └── synthetic_mixtures/           # Generated synthetic mixture audios
│   └── benchmark_results/            # Whisper results on English and Twi mixtures
│   └── librispeech/                  # Librispeech data
│   └── waxal/                        # Waxal data
├── src/
│   └── preprocess.py                 # Audio loading, normalization, text normalization, and WAXAL cache builder
│   └── evaluate_baseline.py
│                                     
├── notebooks/
│   └── test_preprocess.ipynb         # Verify preprocessing on LibriSpeech + WAXAL samples
│   ├── ADL_Generate_Metadata.ipynb   # Generate mix recipe CSVs (Colab)
│   ├── ADL_Generate_Mixtures.ipynb    # Generate synthetic mixed audio files (Colab)
│   └── ADL_Benchmark_Mixtures.ipynb  # Run benchmarking and plot results (Colab)
└── models/
│   └── whisper-small-twi/            # Fine-tuned Twi Whisper model
├── proposal/
├── report/
└── README.md
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and create .venv
uv sync
```

### System dependency

`ffmpeg` is required for audio decoding and is not managed by `uv`:

```bash
# Linux / Google Colab
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Datasets

### English: LibriSpeech

Downloads the `test-clean` and `dev-clean` splits and builds JSONL manifests:

```bash
uv run python data/download_librispeech.py
```

Files are saved to `data/librispeech/`.

### Twi: WAXAL (AkanASR)

Downloads the WAXAL dataset from HuggingFace Hub and saves it to disk in Arrow format:

```bash
uv run python data/download_waxal.py
```

Files are saved to `data/waxal/`.

> **Note:** The WAXAL dataset is stored in HuggingFace's Arrow format, not as loose audio files. Audio arrays are accessed via the `build_waxal_cache` function in `src/preprocess.py`, which loads and decodes the dataset into an in-memory dict keyed by filename. This is necessary because the audio bytes are embedded inside Arrow files and cannot be read directly by `librosa` or `soundfile`.

---

## Preprocessing (`src/preprocess.py`)

All audio and text preprocessing is centralized here so both local scripts and Colab notebooks use the same logic.

### WAXAL Cache
Added for preprocessing WAXAL data.
```python
from src.preprocess import build_waxal_cache

cache = build_waxal_cache("data/waxal")
# cache: dict mapping filename (e.g. "ak_gh_image_0013_....mp3") -> float32 numpy array
```

---

## Notebooks

### Local (`test_preprocess.ipynb`)

Verifies preprocessing on both LibriSpeech and WAXAL samples. It plays audio inline, prints raw vs normalized transcripts, and verifies the normalizer does not mangle Twi charaters

### Colab (`ADL_*.ipynb`)

The three Colab notebooks are in `notebooks/`. Each notebook imports shared utilities from `src/preprocess.py`.

| Notebook | Purpose |
|---|---|
| `ADL_Generate_Metadata` | Generates mix recipe CSVs for LibriSpeech and WAXAL, pairing target and interferer utterances with random SIR levels and overlap ratios |
| `ADL_Generate_Mixtures` | Reads mix recipe CSVs and generates synthetic mixed `.wav` files for both datasets |
| `ADL_Benchmark_Mixtures` | Runs Whisper small inference on all mixed audio files and computes WER, CER, and noise leakage rate across SIR/overlap buckets |

---

## Experiment Configuration

Mixture recipes are generated with the following parameters:

| Parameter | Values |
|---|---|
| SIR levels (dB) | -5, 0, 5 |
| Overlap ratios | 0.1, 0.2, 0.3, 0.4, 0.5, 1.0 |
| Mixes per dataset | 12,500 |
| Baseline rows (clean) | 500 per dataset (overlap = 0.0) |

Baseline rows use the original clean audio with no mixing, providing a reference WER at 0% interference.

---

## Models

| Dataset | Model | Location |
|---|---|---|
| English (LibriSpeech) | `whisper-small` | `openai/whisper-small` via HuggingFace Hub |
| Twi (WAXAL) | `whisper-small-twi` | `TargetTTS/models/whisper-small-twi/final/` on Drive |

Both models use `language="english"` since Twi fine-tuned model was trained with this setting since Whisper does not natively support Twi.

---

## Metrics

| Metric | Description |
|---|---|
| WER | Word Error Rate: Rratio of word-level edits to reference length |
| CER | Character Error Rate: Same at character level |
| Leakage Rate | Fraction of noise transcript words that appear in the prediction but not the target transcript |

All metrics are computed on normalized text. WER > 1.0 is possible when the model produces more words than the reference (insertions).
