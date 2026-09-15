"""
Re-rank Whisper n-best transcripts with the Twi KenLM.

Single-clip CLI. Does not download data by itself; you pass a local
checkpoint or Hub id and a local wav.

Usage:
    python3 -m scripts.rescore_twi_nbest \\
      --model PATH_OR_HUB_ID \\
      --audio some.wav \\
      --lm data/kenlm/waxal_aka_5gram.klm \\
      --k 5 \\
      --lm-weight 0.5 \\
      --ref "optional reference"
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from src.preprocess import TARGET_SAMPLE_RATE, normalize_akan_text

DEFAULT_LM = Path("data/kenlm/waxal_aka_5gram.klm")
DEFAULT_QUERY = Path("/home/ubuntu/kenlm/build/bin/query")
TOTAL_RE = re.compile(r"Total:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def kenlm_query_bin() -> Path:
    env = os.environ.get("KENLM_QUERY")
    return Path(env) if env else DEFAULT_QUERY


def kenlm_sentence_score(lm_path: Path, sentence: str) -> float:
    """Return KenLM log10 sentence score (query's Total:)."""
    query = kenlm_query_bin()
    if not query.is_file() or not os.access(query, os.X_OK):
        raise FileNotFoundError(
            f"KenLM query binary not found or not executable: {query}"
        )
    text = (sentence or "").strip()
    if not text:
        text = "<unk>"
    proc = subprocess.run(
        [str(query), str(lm_path)],
        input=text + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    matches = TOTAL_RE.findall(combined)
    if not matches:
        raise RuntimeError(
            f"Could not parse KenLM Total: from query output.\n"
            f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return float(matches[-1])


def combined_score(
    asr_score: float,
    n_asr_tokens: int,
    kenlm_score: float,
    n_words: int,
    lm_weight: float,
) -> float:
    asr_norm = asr_score / max(n_asr_tokens, 1)
    lm_norm = kenlm_score / max(n_words, 1)
    return asr_norm + lm_weight * lm_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-rank Whisper n-best hypotheses with a Twi KenLM."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local Whisper directory or Hugging Face model id",
    )
    parser.add_argument("--audio", required=True, type=Path, help="16 kHz wav/flac")
    parser.add_argument(
        "--lm",
        type=Path,
        default=DEFAULT_LM,
        help=f"KenLM binary or ARPA (default: {DEFAULT_LM})",
    )
    parser.add_argument("--k", type=int, default=5, help="How many hypotheses to return")
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Beam width (default: 2 * k, rounded up to a multiple of k)",
    )
    parser.add_argument(
        "--diversity-penalty",
        type=float,
        default=1.0,
        help="Diverse beam search penalty; 0 disables diversity groups",
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=0.5,
        help="Weight on length-normalized KenLM score",
    )
    parser.add_argument("--ref", default=None, help="Optional reference transcript")
    parser.add_argument(
        "--force-english-prompt",
        action="store_true",
        help="Force Whisper decoder language=english (old Twi notebook behavior)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or leave unset to auto-detect",
    )
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise SystemExit("error: --k must be >= 1")
    if not args.audio.is_file():
        raise SystemExit(f"error: audio not found: {args.audio}")
    if not args.lm.is_file():
        raise SystemExit(f"error: KenLM file not found: {args.lm}")
    query = kenlm_query_bin()
    if not query.is_file():
        raise SystemExit(f"error: KenLM query binary not found: {query}")
    model_is_local = Path(args.model).exists()
    if ("/" in args.model or args.model.startswith(".")) and not model_is_local:
        # Looks like a filesystem path that is missing.
        if args.model.startswith("/") or args.model.startswith(".") or args.model.endswith(".pt"):
            raise SystemExit(f"error: Whisper model path not found: {args.model}")


def load_audio_array(path: Path):
    from src.preprocess import preprocess_audio

    waveform = preprocess_audio(str(path))
    return waveform.squeeze(0).numpy()


def _token_lens(sequences, processor):
    pad_id = processor.tokenizer.pad_token_id
    lens = []
    for seq in sequences:
        n = int((seq != pad_id).sum().item()) if pad_id is not None else int(seq.numel())
        lens.append(max(n, 1))
    return lens


def _decode_generate(out, processor):
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    if getattr(out, "sequences_scores", None) is not None:
        scores = [float(x) for x in out.sequences_scores.detach().cpu().tolist()]
    else:
        scores = [0.0] * len(texts)
    return texts, scores, _token_lens(out.sequences, processor)


def _merge_unique(texts, scores, token_lens, extra_texts, extra_scores, extra_lens, k: int):
    seen = {normalize_akan_text(t) for t in texts}
    for raw, asr, n_tok in zip(extra_texts, extra_scores, extra_lens):
        key = normalize_akan_text(raw)
        if key in seen:
            continue
        seen.add(key)
        texts.append(raw)
        scores.append(asr)
        token_lens.append(n_tok)
        if len(texts) >= k:
            break
    return texts, scores, token_lens


def generate_nbest(
    model,
    processor,
    audio,
    device,
    k: int,
    force_english: bool,
    num_beams: int | None,
    diversity_penalty: float,
):
    import torch

    inputs = processor(
        audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs["input_features"].to(device)
    attention_mask = torch.ones(
        input_features.shape[0], input_features.shape[-1], device=device
    )

    beams = num_beams if num_beams is not None else max(2 * k, k)
    if beams < k:
        beams = k
    groups = k if (diversity_penalty > 0 and k > 1) else 1
    if groups > 1 and beams % groups != 0:
        beams = ((beams + groups - 1) // groups) * groups

    base_kwargs = {
        "attention_mask": attention_mask,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if force_english:
        base_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    # Always keep a real beam 1-best so KenLM is compared against Whisper, not
    # against a random sample.
    print(f"Beam 1-best: num_beams={beams}", flush=True)
    with torch.no_grad():
        out_one = model.generate(
            input_features,
            **base_kwargs,
            num_beams=beams,
            num_return_sequences=1,
        )
    texts, scores, token_lens = _decode_generate(out_one, processor)

    if k > 1 and groups > 1:
        print(
            f"Diverse beam search: num_beams={beams} groups={groups} "
            f"diversity_penalty={diversity_penalty}",
            flush=True,
        )
        diverse_kwargs = {
            **base_kwargs,
            "num_beams": beams,
            "num_return_sequences": k,
            "num_beam_groups": groups,
            "diversity_penalty": diversity_penalty,
            # transformers>=4.62 moved group beam search out of core.
            "custom_generate": "transformers-community/group-beam-search",
            "trust_remote_code": True,
        }
        try:
            with torch.no_grad():
                out_div = model.generate(input_features, **diverse_kwargs)
            extra_texts, extra_scores, extra_lens = _decode_generate(
                out_div, processor
            )
            texts, scores, token_lens = _merge_unique(
                texts, scores, token_lens, extra_texts, extra_scores, extra_lens, k
            )
        except Exception as exc:
            print(
                f"Diverse beam generate failed ({exc}); "
                "will sample extra hypotheses",
                flush=True,
            )

    if k > 1 and len(texts) < k:
        n_sample = max(2 * k, 8)
        print(
            f"Sampling extra hyps: temperature=0.8 return={n_sample}",
            flush=True,
        )
        with torch.no_grad():
            out_s = model.generate(
                input_features,
                **base_kwargs,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=n_sample,
            )
        extra_texts, extra_scores, extra_lens = _decode_generate(out_s, processor)
        texts, scores, token_lens = _merge_unique(
            texts, scores, token_lens, extra_texts, extra_scores, extra_lens, k
        )

    # Mixed beam + sample scores are not comparable; rank by KenLM only then.
    if any(abs(s) < 1e-12 for s in scores):
        scores = [0.0] * len(texts)

    n_unique = len({normalize_akan_text(t) for t in texts})
    print(
        f"Returned {len(texts)} hypotheses, {n_unique} unique after Akan normalize",
        flush=True,
    )
    return texts, scores, token_lens


def rescore_hypotheses(raw_texts, asr_scores, token_lens, lm_path: Path, lm_weight: float):
    rows = []
    for i, (raw, asr, n_tok) in enumerate(zip(raw_texts, asr_scores, token_lens)):
        cleaned = normalize_akan_text(raw)
        n_words = len(cleaned.split()) if cleaned else 0
        lm_score = kenlm_sentence_score(lm_path, cleaned)
        combo = combined_score(asr, n_tok, lm_score, n_words, lm_weight)
        rows.append(
            {
                "rank_asr": i,
                "raw": raw,
                "cleaned": cleaned,
                "asr_score": asr,
                "n_asr_tokens": n_tok,
                "kenlm_score": lm_score,
                "n_words": n_words,
                "combined": combo,
            }
        )
    rescored = sorted(rows, key=lambda r: r["combined"], reverse=True)
    return rows, rescored


def maybe_wer(ref: str | None, hyp: str) -> float | None:
    if ref is None:
        return None
    import jiwer

    ref_n = normalize_akan_text(ref)
    hyp_n = normalize_akan_text(hyp)
    if not ref_n:
        return None
    return jiwer.wer(ref_n, hyp_n)


def main() -> None:
    args = parse_args()
    require_inputs(args)

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    local = Path(args.model).exists()
    print(f"Loading Whisper: {args.model} (local={local}) on {device}", flush=True)
    processor = WhisperProcessor.from_pretrained(
        args.model, local_files_only=local
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=local
    ).to(device)
    model.eval()

    audio = load_audio_array(args.audio)
    raw_texts, asr_scores, token_lens = generate_nbest(
        model,
        processor,
        audio,
        device,
        args.k,
        args.force_english_prompt,
        args.num_beams,
        args.diversity_penalty,
    )
    asr_order, rescored = rescore_hypotheses(
        raw_texts, asr_scores, token_lens, args.lm, args.lm_weight
    )

    one_best = asr_order[0]
    picked = rescored[0]

    print("\nASR n-best (Whisper order):")
    for row in asr_order:
        print(
            f"  [{row['rank_asr']}] asr={row['asr_score']:.4f} "
            f"lm={row['kenlm_score']:.4f} combo={row['combined']:.4f}\n"
            f"      {row['cleaned']}"
        )

    print("\n1-best (Whisper):")
    print(f"  {one_best['cleaned']}")
    print("Rescored:")
    print(f"  {picked['cleaned']}")
    if picked["cleaned"] == one_best["cleaned"]:
        print("  (KenLM did not change the 1-best)")
    else:
        print("  (KenLM selected a different hypothesis)")

    if args.ref is not None:
        wer_1 = maybe_wer(args.ref, one_best["cleaned"])
        wer_r = maybe_wer(args.ref, picked["cleaned"])
        print("\nWER (Akan-normalized):")
        if wer_1 is not None:
            print(f"  1-best:   {wer_1:.3f}")
        if wer_r is not None:
            print(f"  rescored: {wer_r:.3f}")


if __name__ == "__main__":
    main()
