"""Decode and evaluate Twi Whisper with corrected incremental KenLM fusion.

The three decoding arms are:

* ``vanilla``: unmodified Whisper beam search;
* ``fusion_control``: Whisper with the same per-beam acoustic top-k pruning
  used by fusion, but no LM adjustment;
* ``fusion``: the control plus incremental completed-word KenLM deltas.

Example validation experiment:

    python3 -m scripts.decode_twi_whisper_lm \
      --eval-dir data/kenlm/eval/val_corrected \
      --lm data/kenlm/waxal_aka_train_5gram.klm \
      --calibrate-top-k --grid

Evaluate the locked choice once:

    python3 -m scripts.decode_twi_whisper_lm \
      --eval-dir data/kenlm/eval/test \
      --lm data/kenlm/waxal_aka_train_5gram.klm \
      --evaluate-chosen data/kenlm/eval/val_corrected/corrected_results/chosen_lm_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from src.preprocess import TARGET_SAMPLE_RATE, normalize_akan_text
from src.whisper_kenlm import (
    WhisperAcousticTopKLogitsProcessor,
    WhisperKenLMLogitsProcessor,
    load_kenlm,
)

DEFAULT_MODEL = "waxal-benchmarking/whisper-small-waxal-aka"
DEFAULT_LM = Path("data/kenlm/waxal_aka_train_5gram.klm")
DEFAULT_ALPHA_GRID = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
DEFAULT_CANDIDATE_POOLS = (32, 64, 128)
ARMS = ("vanilla", "fusion_control", "fusion")
LOOP_TRIGRAM_FRAC = 0.40
LOOP_LEN_RATIO = 1.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected incremental KenLM fusion for Twi Whisper."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", type=Path, default=None, help="One 16 kHz clip")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Dir of wav+ref")
    parser.add_argument("--ref", default=None, help="Reference for --audio")
    parser.add_argument("--lm", type=Path, default=DEFAULT_LM)
    parser.add_argument("--arm", choices=ARMS, default="vanilla")
    parser.add_argument("--lm-alpha", type=float, default=0.0)
    parser.add_argument("--lm-beta", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Experiment output directory (default: <eval-dir>/corrected_results)",
    )
    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=0,
        help="Apply the same no-repeat constraint to every compared arm",
    )
    parser.add_argument(
        "--calibrate-top-k",
        action="store_true",
        help="Try candidate pools until fusion_control matches vanilla",
    )
    parser.add_argument(
        "--candidate-pools",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_POOLS),
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Tune beta=0 on the predeclared alpha grid",
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_GRID),
    )
    parser.add_argument(
        "--params-out",
        type=Path,
        default=None,
        help="Locked-parameter JSON path",
    )
    parser.add_argument(
        "--evaluate-chosen",
        type=Path,
        default=None,
        help="Evaluate a locked validation choice on --eval-dir",
    )
    return parser.parse_args()


def load_audio_array(path: Path):
    from src.preprocess import preprocess_audio

    return preprocess_audio(str(path)).squeeze(0).numpy()


def list_eval_clips(eval_dir: Path) -> list[dict]:
    if not eval_dir.is_dir():
        raise SystemExit(f"error: eval dir not found: {eval_dir}")
    clips = []
    for wav in sorted(eval_dir.glob("*.wav")):
        ref_path = wav.with_suffix(".ref.txt")
        if not ref_path.is_file():
            print(f"warning: no reference for {wav.name}; skipping", flush=True)
            continue
        clips.append(
            {
                "id": wav.stem,
                "wav": wav,
                "ref": ref_path.read_text(encoding="utf-8").strip(),
            }
        )
    if not clips:
        raise SystemExit(f"error: no wav+ref pairs in {eval_dir}")
    return clips


def load_model(model_id: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local = Path(model_id).exists()
    print(f"Loading Whisper: {model_id} (local={local}) on {device}", flush=True)
    processor = WhisperProcessor.from_pretrained(model_id, local_files_only=local)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local
    ).to(device)
    model.eval()
    return processor, model


def transcribe(
    model,
    processor,
    audio,
    device: str,
    num_beams: int,
    arm: str = "vanilla",
    lm_model=None,
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
    candidate_top_k: int = 32,
    no_repeat_ngram: int = 0,
) -> tuple[str, list[int]]:
    import torch
    from transformers.generation import LogitsProcessorList

    if arm not in ARMS:
        raise ValueError(f"unknown decoding arm: {arm}")
    if arm == "fusion" and lm_model is None:
        raise ValueError("fusion arm requires a KenLM model")

    inputs = processor(
        audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs["input_features"].to(device)
    attention_mask = torch.ones(
        input_features.shape[0], input_features.shape[-1], device=device
    )
    gen_kwargs = {
        "attention_mask": attention_mask,
        "num_beams": num_beams,
        "do_sample": False,
        "temperature": 0.0,
        "return_timestamps": False,
        "max_new_tokens": 224,
    }

    if arm == "fusion_control":
        logits_processor = WhisperAcousticTopKLogitsProcessor(candidate_top_k)
        gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    elif arm == "fusion":
        logits_processor = WhisperKenLMLogitsProcessor(
            tokenizer=processor.tokenizer,
            lm_model=lm_model,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
            top_k=candidate_top_k,
            lm_token_threshold=1,
        )
        gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    if no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram)

    with torch.no_grad():
        ids = model.generate(input_features, **gen_kwargs)
    text = processor.batch_decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text, ids[0].detach().cpu().tolist()


def hyp_is_loop(hyp: str, ref: str | None) -> bool:
    hyp_words = normalize_akan_text(hyp).split()
    ref_words = normalize_akan_text(ref or "").split()
    if ref_words and len(hyp_words) > LOOP_LEN_RATIO * len(ref_words):
        return True
    if len(hyp_words) < 3:
        return False
    trigrams = [tuple(hyp_words[i : i + 3]) for i in range(len(hyp_words) - 2)]
    top_count = Counter(trigrams).most_common(1)[0][1]
    return (top_count / len(trigrams)) > LOOP_TRIGRAM_FRAC


def _wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(normalize_akan_text(ref), normalize_akan_text(hyp)))


def _lm_metrics(text: str, lm_model) -> dict:
    normalized = normalize_akan_text(text)
    words = normalized.split()
    if not words or lm_model is None:
        return {
            "log10": None,
            "log10_per_word": None,
            "words": len(words),
            "oov": None,
            "oov_rate": None,
        }
    score = float(lm_model.score(normalized, bos=True, eos=True))
    oov = sum(1 for word in words if word not in lm_model)
    return {
        "log10": score,
        "log10_per_word": score / len(words),
        "words": len(words),
        "oov": oov,
        "oov_rate": oov / len(words),
    }


def _is_clean_twi(ref: str) -> tuple[bool, list[str]]:
    from scripts.prepare_waxal_eval_clips import english_hits

    hits = english_hits(ref)
    return len(hits) < 2, hits


def run_eval_dir(
    processor,
    model,
    device: str,
    clips: list[dict],
    lm_model,
    *,
    arm: str,
    lm_alpha: float,
    lm_beta: float,
    num_beams: int,
    candidate_top_k: int,
    no_repeat_ngram: int,
    audio_cache: dict[str, object] | None = None,
) -> list[dict]:
    rows = []
    audio_cache = audio_cache if audio_cache is not None else {}
    for index, clip in enumerate(clips, start=1):
        print(
            f"[{index}/{len(clips)}] {clip['id']} arm={arm} "
            f"alpha={lm_alpha:g} beta={lm_beta:g} k={candidate_top_k}",
            flush=True,
        )
        audio = audio_cache.get(clip["id"])
        if audio is None:
            audio = load_audio_array(clip["wav"])
            audio_cache[clip["id"]] = audio
        raw, token_ids = transcribe(
            model,
            processor,
            audio,
            device,
            num_beams,
            arm=arm,
            lm_model=lm_model,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
            candidate_top_k=candidate_top_k,
            no_repeat_ngram=no_repeat_ngram,
        )
        hyp = normalize_akan_text(raw)
        ref = normalize_akan_text(clip["ref"])
        hyp_lm = _lm_metrics(hyp, lm_model)
        ref_lm = _lm_metrics(ref, lm_model)
        clean_twi, english_hits = _is_clean_twi(ref)
        row = {
            "id": clip["id"],
            "wav": str(clip["wav"]),
            "arm": arm,
            "lm_alpha": lm_alpha if arm == "fusion" else None,
            "lm_beta": lm_beta if arm == "fusion" else None,
            "candidate_top_k": candidate_top_k if arm != "vanilla" else None,
            "no_repeat_ngram": no_repeat_ngram,
            "raw": raw,
            "hyp": hyp,
            "ref": ref,
            "token_ids": token_ids,
            "output_tokens": len(token_ids),
            "output_words": len(hyp.split()),
            "wer": _wer(ref, hyp),
            "looped": hyp_is_loop(hyp, ref),
            "clean_twi": clean_twi,
            "english_hint_hits": english_hits,
            "hyp_lm": hyp_lm,
            "ref_lm": ref_lm,
            "reference_better_lm_per_word": (
                ref_lm["log10_per_word"] is not None
                and hyp_lm["log10_per_word"] is not None
                and ref_lm["log10_per_word"] > hyp_lm["log10_per_word"]
            ),
        }
        rows.append(row)
        flag = " LOOP" if row["looped"] else ""
        print(f"  WER={row['wer']:.3f}{flag}  {hyp[:120]}", flush=True)

    stats = summarize_rows(rows)
    print(
        f"  corpus WER={stats['corpus_wer']:.4f}; "
        f"mean={stats['mean_wer']:.4f}; median={stats['median_wer']:.4f}; "
        f"loops={stats['n_looped']}; OOV={stats['hyp_oov_rate']:.4f}",
        flush=True,
    )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    import jiwer

    if not rows:
        return {"n": 0}
    refs = [row["ref"] for row in rows]
    hyps = [row["hyp"] for row in rows]
    wers = [float(row["wer"]) for row in rows]
    hyp_words = sum(row["hyp_lm"]["words"] for row in rows)
    ref_words = sum(row["ref_lm"]["words"] for row in rows)
    hyp_oov_values = [row["hyp_lm"]["oov"] for row in rows]
    ref_oov_values = [row["ref_lm"]["oov"] for row in rows]
    hyp_scores = [row["hyp_lm"]["log10"] for row in rows]
    ref_scores = [row["ref_lm"]["log10"] for row in rows]
    have_lm = all(value is not None for value in hyp_scores + ref_scores)
    return {
        "n": len(rows),
        "corpus_wer": float(jiwer.wer(refs, hyps)),
        "mean_wer": sum(wers) / len(wers),
        "median_wer": float(median(wers)),
        "n_looped": sum(bool(row["looped"]) for row in rows),
        "looped_ids": [row["id"] for row in rows if row["looped"]],
        "hyp_words": hyp_words,
        "ref_words": ref_words,
        "mean_output_words": hyp_words / len(rows),
        "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
        "hyp_oov": sum(hyp_oov_values) if have_lm else None,
        "ref_oov": sum(ref_oov_values) if have_lm else None,
        "hyp_oov_rate": (
            sum(hyp_oov_values) / hyp_words if have_lm and hyp_words else None
        ),
        "ref_oov_rate": (
            sum(ref_oov_values) / ref_words if have_lm and ref_words else None
        ),
        "hyp_lm_log10_per_word": (
            sum(hyp_scores) / hyp_words if have_lm and hyp_words else None
        ),
        "ref_lm_log10_per_word": (
            sum(ref_scores) / ref_words if have_lm and ref_words else None
        ),
        "reference_better_lm_count": sum(
            bool(row["reference_better_lm_per_word"]) for row in rows
        ),
    }


def summary_bundle(rows: list[dict]) -> dict:
    clean_rows = [row for row in rows if row["clean_twi"]]
    return {
        "all": summarize_rows(rows),
        "clean_twi": summarize_rows(clean_rows),
    }


def compare_rows(base_rows: list[dict], candidate_rows: list[dict]) -> dict:
    base = {row["id"]: row for row in base_rows}
    improved = []
    worsened = []
    changed = []
    token_mismatches = []
    for row in candidate_rows:
        old = base[row["id"]]
        if row["hyp"] != old["hyp"]:
            changed.append(row["id"])
        if row["token_ids"] != old["token_ids"]:
            token_mismatches.append(row["id"])
        if row["wer"] < old["wer"] - 1e-12:
            improved.append(row["id"])
        elif row["wer"] > old["wer"] + 1e-12:
            worsened.append(row["id"])
    return {
        "n_changed": len(changed),
        "changed_ids": changed,
        "n_improved": len(improved),
        "improved_ids": improved,
        "n_worsened": len(worsened),
        "worsened_ids": worsened,
        "n_token_mismatches": len(token_mismatches),
        "token_mismatch_ids": token_mismatches,
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {path}", flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path}", flush=True)


def _run_arm(
    args,
    processor,
    model,
    device,
    clips,
    lm_model,
    audio_cache,
    *,
    arm,
    alpha=0.0,
    top_k=None,
):
    return run_eval_dir(
        processor,
        model,
        device,
        clips,
        lm_model,
        arm=arm,
        lm_alpha=alpha,
        lm_beta=0.0,
        num_beams=args.num_beams,
        candidate_top_k=top_k or args.candidate_top_k,
        no_repeat_ngram=args.no_repeat_ngram,
        audio_cache=audio_cache,
    )


def calibrate_candidate_pool(
    args,
    processor,
    model,
    device,
    clips,
    lm_model,
    out_dir,
    audio_cache,
) -> dict:
    print("\n=== vanilla reference ===", flush=True)
    vanilla = _run_arm(
        args, processor, model, device, clips, lm_model, audio_cache, arm="vanilla"
    )
    write_jsonl(out_dir / "validation_vanilla.jsonl", vanilla)
    write_json(
        out_dir / "heldout_lm_diagnostics.json",
        {
            "lm": str(args.lm),
            "split": "validation selected before decoding",
            "vanilla": summary_bundle(vanilla),
            "per_clip": [
                {
                    "id": row["id"],
                    "ref_lm": row["ref_lm"],
                    "hyp_lm": row["hyp_lm"],
                    "reference_better_lm_per_word": row[
                        "reference_better_lm_per_word"
                    ],
                }
                for row in vanilla
            ],
        },
    )

    trials = []
    chosen_k = None
    chosen_control = None
    for top_k in args.candidate_pools:
        print(f"\n=== fusion control candidate top-k={top_k} ===", flush=True)
        control = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm="fusion_control",
            top_k=top_k,
        )
        comparison = compare_rows(vanilla, control)
        exact = comparison["n_token_mismatches"] == 0
        trials.append(
            {
                "candidate_top_k": top_k,
                "exact_token_match": exact,
                "comparison_to_vanilla": comparison,
                "summary": summary_bundle(control),
            }
        )
        write_jsonl(out_dir / f"validation_control_k{top_k}.jsonl", control)
        chosen_k = top_k
        chosen_control = control
        if exact:
            break

    print(f"\n=== alpha=0 fusion identity control, top-k={chosen_k} ===", flush=True)
    fusion_zero = _run_arm(
        args,
        processor,
        model,
        device,
        clips,
        lm_model,
        audio_cache,
        arm="fusion",
        alpha=0.0,
        top_k=chosen_k,
    )
    zero_comparison = compare_rows(chosen_control, fusion_zero)
    if zero_comparison["n_token_mismatches"]:
        raise RuntimeError(
            "alpha=beta=0 fusion did not exactly match fusion_control: "
            f"{zero_comparison['token_mismatch_ids']}"
        )
    write_jsonl(out_dir / f"validation_fusion_alpha0_k{chosen_k}.jsonl", fusion_zero)
    result = {
        "candidate_top_k": chosen_k,
        "matched_vanilla": trials[-1]["exact_token_match"],
        "trials": trials,
        "alpha_zero_matches_control": True,
        "alpha_zero_comparison": zero_comparison,
        "no_repeat_ngram": args.no_repeat_ngram,
    }
    write_json(out_dir / "candidate_pool_calibration.json", result)
    return {
        "report": result,
        "vanilla": vanilla,
        "control": chosen_control,
        "fusion_zero": fusion_zero,
    }


def tune_alpha_grid(
    args,
    processor,
    model,
    device,
    clips,
    lm_model,
    out_dir,
    audio_cache,
    calibration=None,
) -> dict:
    top_k = (
        calibration["report"]["candidate_top_k"]
        if calibration is not None
        else args.candidate_top_k
    )
    if calibration is None:
        vanilla = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm="vanilla",
        )
        control = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm="fusion_control",
            top_k=top_k,
        )
        fusion_zero = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm="fusion",
            alpha=0.0,
            top_k=top_k,
        )
        if compare_rows(control, fusion_zero)["n_token_mismatches"]:
            raise RuntimeError("alpha=0 fusion did not match fusion_control")
    else:
        vanilla = calibration["vanilla"]
        control = calibration["control"]
        fusion_zero = calibration["fusion_zero"]

    control_stats = summary_bundle(control)
    settings = []
    rows_by_alpha = {}
    for alpha in args.alpha_grid:
        print(f"\n=== alpha grid: alpha={alpha:g}, beta=0 ===", flush=True)
        if alpha == 0.0:
            rows = fusion_zero
        else:
            rows = _run_arm(
                args,
                processor,
                model,
                device,
                clips,
                lm_model,
                audio_cache,
                arm="fusion",
                alpha=alpha,
                top_k=top_k,
            )
        rows_by_alpha[alpha] = rows
        stats = summary_bundle(rows)
        comparison = compare_rows(control, rows)
        all_stats = stats["all"]
        control_all = control_stats["all"]
        eligible = (
            alpha > 0.0
            and all_stats["corpus_wer"] < control_all["corpus_wer"] - 1e-12
            and comparison["n_improved"] > comparison["n_worsened"]
            and all_stats["n_looped"] == 0
            and all_stats["hyp_oov_rate"]
            <= control_all["hyp_oov_rate"] + 0.02 + 1e-12
        )
        settings.append(
            {
                "alpha": alpha,
                "beta": 0.0,
                "eligible": eligible,
                "summary": stats,
                "comparison_to_fusion_control": comparison,
                "safety": {
                    "lower_corpus_wer": all_stats["corpus_wer"]
                    < control_all["corpus_wer"] - 1e-12,
                    "more_improved_than_worsened": comparison["n_improved"]
                    > comparison["n_worsened"],
                    "no_loops": all_stats["n_looped"] == 0,
                    "oov_increase_pp": 100.0
                    * (
                        all_stats["hyp_oov_rate"]
                        - control_all["hyp_oov_rate"]
                    ),
                    "oov_within_2pp": all_stats["hyp_oov_rate"]
                    <= control_all["hyp_oov_rate"] + 0.02 + 1e-12,
                },
            }
        )
        write_jsonl(
            out_dir / f"validation_fusion_alpha{alpha:g}.jsonl",
            rows,
        )

    eligible_settings = [setting for setting in settings if setting["eligible"]]
    if eligible_settings:
        selected = min(
            eligible_settings,
            key=lambda setting: (
                setting["summary"]["all"]["corpus_wer"],
                setting["alpha"],
            ),
        )
        chosen = "fusion"
        chosen_alpha = selected["alpha"]
    else:
        selected = None
        chosen = "no_lm"
        chosen_alpha = None

    params = {
        "chosen": chosen,
        "lm_alpha": chosen_alpha,
        "lm_beta": 0.0 if chosen == "fusion" else None,
        "candidate_top_k": top_k,
        "num_beams": args.num_beams,
        "no_repeat_ngram": args.no_repeat_ngram,
        "lm": str(args.lm),
        "validation_n": len(clips),
        "fusion_control_matches_vanilla": (
            compare_rows(vanilla, control)["n_token_mismatches"] == 0
        ),
        "vanilla": summary_bundle(vanilla),
        "fusion_control": control_stats,
        "settings": settings,
        "selection_rule": (
            "lower corpus WER than fusion_control; more clips improved than "
            "worsened; zero loops; OOV increase <= 2 percentage points"
        ),
    }
    params_out = args.params_out or (out_dir / "chosen_lm_params.json")
    write_json(params_out, params)
    print(f"Locked chosen={chosen} alpha={chosen_alpha}", flush=True)
    return params


def evaluate_locked_choice(
    args,
    processor,
    model,
    device,
    clips,
    lm_model,
    out_dir,
    audio_cache,
) -> dict:
    params = json.loads(args.evaluate_chosen.read_text(encoding="utf-8"))
    top_k = int(params["candidate_top_k"])
    no_repeat = int(params.get("no_repeat_ngram", 0))
    if args.no_repeat_ngram not in (0, no_repeat):
        raise SystemExit(
            "error: test no-repeat setting must equal the locked validation setting"
        )
    args.no_repeat_ngram = no_repeat

    print("\n=== frozen test: vanilla ===", flush=True)
    vanilla = _run_arm(
        args, processor, model, device, clips, lm_model, audio_cache, arm="vanilla"
    )
    print("\n=== frozen test: fusion control ===", flush=True)
    control = _run_arm(
        args,
        processor,
        model,
        device,
        clips,
        lm_model,
        audio_cache,
        arm="fusion_control",
        top_k=top_k,
    )
    fusion = None
    if params["chosen"] == "fusion":
        print("\n=== frozen test: locked fusion ===", flush=True)
        fusion = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm="fusion",
            alpha=float(params["lm_alpha"]),
            top_k=top_k,
        )

    arms = {
        "vanilla": summary_bundle(vanilla),
        "fusion_control": summary_bundle(control),
        "fusion": summary_bundle(fusion) if fusion is not None else None,
    }
    report = {
        "locked_params": params,
        "n_test": len(clips),
        "arms": arms,
        "fusion_control_vs_vanilla": compare_rows(vanilla, control),
        "fusion_vs_control": compare_rows(control, fusion)
        if fusion is not None
        else None,
    }
    write_jsonl(out_dir / "test_vanilla.jsonl", vanilla)
    write_jsonl(out_dir / "test_fusion_control.jsonl", control)
    if fusion is not None:
        write_jsonl(out_dir / "test_fusion_locked.jsonl", fusion)
    write_json(out_dir / "test_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    if args.audio is None and args.eval_dir is None:
        raise SystemExit("error: pass --audio or --eval-dir")
    if args.audio is not None and not args.audio.is_file():
        raise SystemExit(f"error: audio not found: {args.audio}")
    if (args.calibrate_top_k or args.grid or args.evaluate_chosen) and not args.eval_dir:
        raise SystemExit("error: experiments require --eval-dir")
    if args.num_beams < 1 or args.candidate_top_k < 1:
        raise SystemExit("error: beam and candidate counts must be positive")

    need_lm = (
        args.arm == "fusion"
        or args.calibrate_top_k
        or args.grid
        or args.evaluate_chosen is not None
    )
    if need_lm and not args.lm.is_file():
        raise SystemExit(f"error: KenLM file not found: {args.lm}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model(args.model, device)
    lm_model = load_kenlm(args.lm) if args.lm.is_file() else None
    if lm_model is not None:
        print(f"Loaded KenLM: {args.lm}", flush=True)

    if args.eval_dir is not None:
        clips = list_eval_clips(args.eval_dir)
        out_dir = args.out_dir or (args.eval_dir / "corrected_results")
        audio_cache = {}
        if args.evaluate_chosen is not None:
            evaluate_locked_choice(
                args,
                processor,
                model,
                device,
                clips,
                lm_model,
                out_dir,
                audio_cache,
            )
            return
        calibration = None
        if args.calibrate_top_k:
            calibration = calibrate_candidate_pool(
                args,
                processor,
                model,
                device,
                clips,
                lm_model,
                out_dir,
                audio_cache,
            )
        if args.grid:
            tune_alpha_grid(
                args,
                processor,
                model,
                device,
                clips,
                lm_model,
                out_dir,
                audio_cache,
                calibration=calibration,
            )
        if args.calibrate_top_k or args.grid:
            return
        rows = _run_arm(
            args,
            processor,
            model,
            device,
            clips,
            lm_model,
            audio_cache,
            arm=args.arm,
            alpha=args.lm_alpha,
            top_k=args.candidate_top_k,
        )
        out = args.out or (out_dir / f"{args.arm}_results.jsonl")
        write_jsonl(out, rows)
        return

    audio = load_audio_array(args.audio)
    raw, _ = transcribe(
        model,
        processor,
        audio,
        device,
        args.num_beams,
        arm=args.arm,
        lm_model=lm_model,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
        candidate_top_k=args.candidate_top_k,
        no_repeat_ngram=args.no_repeat_ngram,
    )
    print("Raw:", raw)
    print("Hyp:", normalize_akan_text(raw))
    if args.ref is not None:
        print("Ref:", normalize_akan_text(args.ref))
        print(f"WER: {_wer(args.ref, raw):.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
