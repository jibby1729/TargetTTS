"""Incremental in-beam KenLM fusion for Hugging Face Whisper.

The beam accumulates one conditional KenLM contribution per completed word:

    Q = log P_whisper + alpha * log P_kenlm + beta * word_count

KenLM returns log10 scores while Whisper uses natural-log probabilities, so
KenLM deltas are multiplied by ln(10).  Potential differences make the score
telescoping: earlier words are never counted again at later decoding steps.
"""

from __future__ import annotations

import math
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import kenlm
import torch
from transformers.generation import LogitsProcessor

from src.preprocess import normalize_akan_text


def load_kenlm(path: str | Path) -> kenlm.Model:
    return kenlm.Model(str(path))


class KenLMScorer(Protocol):
    """The subset of ``kenlm.Model`` used by the fusion processor."""

    def score(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        ...


@dataclass(frozen=True)
class CompletedWordPotential:
    """KenLM potential for all words known to be complete in a token prefix."""

    text: str
    log10_score: float
    word_count: int
    eos: bool


class WhisperAcousticTopKLogitsProcessor(LogitsProcessor):
    """Keep the same per-beam acoustic candidate pool used by LM fusion."""

    def __init__(self, top_k: int):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = int(top_k)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        k = min(self.top_k, scores.shape[-1])
        top_vals, top_inds = torch.topk(scores, k, dim=-1)
        pruned = torch.full_like(scores, float("-inf"))
        return pruned.scatter(1, top_inds, top_vals)


class WhisperKenLMLogitsProcessor(WhisperAcousticTopKLogitsProcessor):
    """Apply incremental KenLM potential deltas to top acoustic candidates."""

    def __init__(
        self,
        tokenizer,
        lm_model: KenLMScorer | str | Path,
        lm_alpha: float = 0.5,
        lm_beta: float = 0.0,
        top_k: int = 32,
        lm_token_threshold: int = 1,
        word_closing_punctuation: str | None = None,
    ):
        super().__init__(top_k=top_k)
        self.tokenizer = tokenizer
        self.special_token_ids = set(tokenizer.all_special_ids)
        self.eos_token_id = tokenizer.eos_token_id
        self.lm_model = (
            load_kenlm(lm_model)
            if isinstance(lm_model, (str, Path))
            else lm_model
        )
        self.lm_alpha = float(lm_alpha)
        self.lm_beta = float(lm_beta)
        self.lm_token_threshold = int(lm_token_threshold)
        self.word_closing_punctuation = set(
            word_closing_punctuation
            if word_closing_punctuation is not None
            else string.punctuation
        )
        self._score_cache: dict[tuple[str, bool], float] = {}
        self._potential_cache: dict[tuple[int, ...], CompletedWordPotential] = {}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Return pruned acoustic scores plus local LM/word-count deltas."""
        if self.lm_alpha == 0.0 and self.lm_beta == 0.0:
            return super().__call__(input_ids, scores)

        k = min(self.top_k, scores.shape[-1])
        top_vals, top_inds = torch.topk(scores, k, dim=-1)
        adjusted = torch.full_like(scores, float("-inf"))

        for j in range(scores.shape[0]):
            prefix_ids = input_ids[j].tolist()
            before = self.completed_word_potential(prefix_ids)

            for i in range(k):
                token_id = int(top_inds[j, i].item())
                after = self.completed_word_potential(prefix_ids + [token_id])
                lm_delta_nat = math.log(10.0) * (
                    after.log10_score - before.log10_score
                )
                word_delta = after.word_count - before.word_count
                adjustment = (
                    self.lm_alpha * lm_delta_nat
                    + self.lm_beta * word_delta
                )
                adjusted[j, token_id] = top_vals[j, i] + adjustment
        return adjusted

    def completed_word_potential(
        self, sequence_ids: list[int] | tuple[int, ...]
    ) -> CompletedWordPotential:
        """Score only words completed by whitespace, punctuation, or EOS."""
        key_ids = tuple(int(token_id) for token_id in sequence_ids)
        cached = self._potential_cache.get(key_ids)
        if cached is not None:
            return cached

        content_count = sum(
            token_id not in self.special_token_ids for token_id in key_ids
        )
        if content_count < self.lm_token_threshold:
            potential = CompletedWordPotential("", 0.0, 0, False)
            self._potential_cache[key_ids] = potential
            return potential

        eos = bool(key_ids) and key_ids[-1] == self.eos_token_id
        decoded = self.tokenizer.decode(
            key_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        completed = self._completed_text(decoded, eos=eos)
        normalized = normalize_akan_text(completed)
        word_count = len(normalized.split()) if normalized else 0

        if not normalized:
            score = 0.0
            eos = False
        else:
            score_key = (normalized, eos)
            score = self._score_cache.get(score_key)
            if score is None:
                score = float(
                    self.lm_model.score(normalized, bos=True, eos=eos)
                )
                self._score_cache[score_key] = score

        potential = CompletedWordPotential(
            text=normalized,
            log10_score=score,
            word_count=word_count,
            eos=eos,
        )
        self._potential_cache[key_ids] = potential
        return potential

    def _completed_text(self, decoded: str, eos: bool) -> str:
        if not decoded:
            return ""
        if eos or decoded[-1].isspace():
            return decoded
        if (
            decoded[-1] in self.word_closing_punctuation
            or unicodedata.category(decoded[-1]).startswith("P")
        ):
            return decoded
        # The last normalized word is still unfinished. Dropping it after
        # normalization also handles punctuation followed by a word with no
        # intervening space (for example ``asɛm.fo``).
        words = normalize_akan_text(decoded).split()
        return " ".join(words[:-1])
