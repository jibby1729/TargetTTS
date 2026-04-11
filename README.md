# TargetTTS: Target-Speaker Speech-to-Text in Overlapping Speech

ECE1508 (Applied Deep Learning) course project. Given a multi-speaker audio mixture and a short enrollment clip of the target speaker, transcribe only the target speaker.

**Pipeline:** Enrollment clip → ECAPA-TDNN (frozen) → speaker embedding → Extraction network (trained) → time-frequency mask → masked spectrogram → iSTFT → Whisper (frozen) → transcript.

---

## Project Structure

```
TargetTTS/
├── src/
│   ├── extraction_net.py       # Extraction network architecture, STFT/iSTFT utilities
│   ├── speaker_encoder.py      # ECAPA-TDNN speaker embedding wrapper
│   ├── preprocess.py           # Audio loading (16kHz, peak-normalized), text normalization, WAXAL cache
│   └── dataset.py              # MixtureDataset and pad_collate for DataLoader
├── scripts/
│   ├── generate_recipes.py     # Generate mixture recipe CSVs
│   ├── generate_mixtures.py    # Generate synthetic 2-speaker WAV mixtures
│   ├── prepare_training.py     # Train/val/test splits + precompute speaker embeddings
│   ├── train_extraction.py     # Train the extraction network
│   ├── evaluate_wer.py         # Evaluate WER with and without extraction
│   └── plot_wer.py             # Plot WER vs overlap ratio from benchmark results
├── data/
│   ├── download_librispeech.py # Download LibriSpeech + build JSONL manifests
│   ├── download_waxal.py       # Download WAXAL (Twi) dataset from HuggingFace Hub
│   ├── mixture_recipes/        # Generated mix recipe CSVs (gitignored)
│   ├── synthetic_mixtures/     # Generated synthetic mixture WAVs (gitignored)
│   ├── benchmark_results/      # Whisper benchmark results for English and Twi
│   ├── librispeech/            # LibriSpeech data (gitignored)
│   └── waxal/                  # WAXAL data (gitignored)
├── notebooks/
│   ├── demo_extraction.ipynb         # Demo: full pipeline on a single example
│   ├── ADL_Generate_Metadata.ipynb   # Generate mix recipes (Colab)
│   ├── ADL_Generate_Mixtures.ipynb   # Generate synthetic mixtures (Colab)
│   ├── ADL_Benchmark_Mixtures.ipynb  # Benchmark Whisper on mixtures (Colab)
│   ├── preview_mixtures.ipynb        # Listen to mixtures and view spectrograms
│   └── test_preprocess.ipynb         # Verify audio and text preprocessing
├── report/
│   ├── final_report.tex        # Final report (NeurIPS format)
│   └── references.bib          # Bibliography
├── results/                    # Evaluation results
├── checkpoints/                # Trained model checkpoints (gitignored)
├── pyproject.toml              # Python dependencies
├── uv.lock                     # Locked dependency versions
└── README.md
```

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package management instead of `requirements.txt` or `environment.yaml`. All dependencies are declared in `pyproject.toml` and pinned in `uv.lock`, which together provide a fully reproducible environment. `uv` resolves, installs, and locks dependencies in a single command and is significantly faster than pip. See the [uv documentation](https://docs.astral.sh/uv/) for more details.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and create .venv
uv sync
```

### GPU / CUDA compatibility

By default, `pyproject.toml` pulls PyTorch from the **CUDA 12.8** wheel index
(`cu128`), which supports all NVIDIA GPUs from Maxwell (sm_50) through Blackwell
(sm_120, e.g. RTX 5090). This requires an NVIDIA driver that supports CUDA 12.8+
(driver ≥ 570).

If you need a different CUDA version, edit the index in `pyproject.toml`:

```toml
# Example: switch to CUDA 12.6
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

For **CPU-only** (no NVIDIA GPU):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then re-run `uv sync`.

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

## Data Pipeline

All data directories are gitignored. Run these steps in order on a new machine:

### 1. Generate mixture recipes and audio

```bash
# Generate recipe CSVs describing which utterances to mix
uv run python -m scripts.generate_recipes --num-mixes 12500 --sir-levels 0 5

# Generate synthetic 2-speaker WAV mixtures (~8.5 GB)
uv run python -m scripts.generate_mixtures
```

Mixtures are generated at SIR levels of 0, 5, and 10 dB with overlap ratios of 0.1, 0.2, 0.3, 0.4, 0.5, and 1.0. Each mixture produces three files: `{mix_id}.wav` (mixture), `{mix_id}_target.wav` (aligned clean target), `{mix_id}_interf.wav` (aligned clean interferer).

### 2. Prepare training data

```bash
uv run python -m scripts.prepare_training
```

Creates `data/prepared_training/` with train/val/test splits (10,000/1,250/1,250) and 2,588 precomputed ECAPA-TDNN speaker embeddings.

---

## Demo

Open `notebooks/demo_extraction.ipynb` to see the full extraction pipeline in action. It runs on a single hard example (2 speakers, 0 dB SIR, 50% overlap) and walks through each step:

1. Listen to the mixture (two speakers talking over each other)
2. Listen to the clean target (what the target speaker sounds like alone)
3. Whisper on the raw mixture (baseline) -- 91% WER
4. Listen to the enrollment clip (different utterance, same speaker)
5. Run extraction: ECAPA-TDNN embedding -> mask prediction -> iSTFT reconstruction
6. Whisper on the extracted audio -- 9% WER

Requires the trained checkpoint at `checkpoints/hybrid_alpha_0.1/best_model.pt` and the synthetic mixtures in `data/synthetic_mixtures/`.

---

## Training

```bash
uv run python -m scripts.train_extraction \
    --batch-size 32 --num-workers 4 --epochs 30 --lr 1e-3 \
    --hybrid-alpha 0.1 --power 0.3
```

**Key flags:**
- `--hybrid-alpha 0.1` — Hybrid loss: 10% oracle mask MSE + 90% power-law reconstruction loss
- `--power 0.3` — Power-law compression exponent (compresses dynamic range in both loss and CNN input)
- `--patience 7` — Early stopping if validation loss doesn't improve for 7 epochs

Best checkpoint is saved to `checkpoints/best_model.pt`. Training prints train loss, val loss, val reconstruction loss, val mask MSE, and learning rate each epoch.

---

## Evaluation

```bash
# Evaluate with extraction vs baseline
uv run python -m scripts.evaluate_wer \
    --checkpoint checkpoints/best_model.pt

# Evaluate on specific conditions
uv run python -m scripts.evaluate_wer \
    --checkpoint checkpoints/best_model.pt \
    --overlap-ratios 0.5 1.0 --sir-levels 0

# Baseline only (no extraction)
uv run python -m scripts.evaluate_wer
```

The evaluation script runs both baseline (raw mixture → Whisper) and extraction (mixture → extraction network → Whisper) on the test split and outputs:
- Per-sample CSVs with WER, CER, and leakage rate
- Aggregated CSVs grouped by SIR level and overlap ratio
- Comparison plot (extraction vs baseline)

---

## Architecture

### Extraction Network (`src/extraction_net.py`)

The only trained component. 9.3M parameters.

- **Input:** Magnitude spectrogram (B, 257, T) + speaker embedding (B, 192)
- **CNN encoder:** 8 Conv2D layers (VoiceFilter architecture). Layers 1-2 are separable (frequency-only, time-only). Layers 3-7 use (5,5) kernels with time-only dilation (1, 2, 4, 8, 16). Layer 8 is a 1x1 bottleneck (64 → 8 channels).
- **Speaker conditioning:** CNN output flattened per frame, speaker embedding concatenated
- **BiLSTM:** 1 layer, 400 hidden per direction
- **Mask predictor:** FC(800→600) + ReLU → FC(600→257) + sigmoid → mask in [0,1]

### Frozen Components

| Component | Model | Purpose |
|---|---|---|
| Speaker encoder | ECAPA-TDNN (SpeechBrain) | 192-dim speaker embedding from enrollment clip |
| ASR (English) | Whisper-tiny (39M params) | Transcription of extracted audio |
| ASR (Twi) | Whisper-small fine-tuned on WAXAL | Transcription for Twi experiments |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `demo_extraction.ipynb` | **Demo:** full extraction pipeline on a single example with audio playback and WER comparison |
| `preview_mixtures.ipynb` | Listen to enrollment clips, clean targets, interferers, and mixtures; view spectrograms |
| `test_preprocess.ipynb` | Verify audio and text preprocessing on LibriSpeech and WAXAL samples |
| `ADL_Generate_Metadata.ipynb` | Generate mix recipe CSVs for LibriSpeech and WAXAL (Colab) |
| `ADL_Generate_Mixtures.ipynb` | Generate synthetic mixed audio files (Colab) |
| `ADL_Benchmark_Mixtures.ipynb` | Benchmark Whisper on mixtures, compute WER/CER/leakage (Colab) |

---

## Metrics

| Metric | Description |
|---|---|
| WER | Word Error Rate: ratio of word-level edits (substitutions + deletions + insertions) to reference length |
| CER | Character Error Rate: same at character level |
| Leakage Rate | Fraction of interferer-only words that appear in the prediction |

All metrics are computed on normalized text (lowercased, punctuation removed, contractions expanded). WER > 1.0 is possible when the model produces more words than the reference.
