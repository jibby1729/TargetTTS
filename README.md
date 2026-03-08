# TargetTTS
ECE1508 Winter course project on isolating target speaker's text transcript given overlapping speech.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and create .venv
uv sync
```

## Getting Started

1. **Download LibriSpeech** — downloads `test-clean` and `dev-clean` splits and builds JSONL manifests:
   ```bash
   uv run python data/download_librispeech.py
   ```

2. **Test preprocessing** — open `notebooks/test_preprocess.ipynb` to listen to samples and view raw vs normalized transcripts.

## Datasets

- **English:** [LibriSpeech](https://www.openslr.org/12) — clean read English speech corpus

## Project Structure

```
TargetTTS/
├── data/
│   └── download_librispeech.py   # download + manifest generation
├── src/
│   └── preprocess.py             # audio loading + text normalization
├── notebooks/
│   └── test_preprocess.ipynb     # listen to samples, view transcripts
├── proposal/
└── README.md
```
