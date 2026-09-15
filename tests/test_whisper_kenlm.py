import math

import kenlm
import pytest
import torch

from src.whisper_kenlm import (
    WhisperAcousticTopKLogitsProcessor,
    WhisperKenLMLogitsProcessor,
)


class TinyTokenizer:
    eos_token_id = 99
    all_special_ids = [99]
    pieces = {
        1: "m",
        2: "ben",
        3: "yinfo",
        4: " n",
        5: "son",
        6: ".",
        7: "ɛ",
        8: "na",
    }

    def decode(
        self,
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        return "".join(
            self.pieces.get(int(token_id), "")
            for token_id in ids
            if not skip_special_tokens or int(token_id) not in self.all_special_ids
        )


class WordCountLM:
    """A deterministic log10 LM: -1 per word and -0.25 for EOS."""

    def score(self, sentence, bos=True, eos=True):
        assert bos is True
        return -float(len(sentence.split())) - (0.25 if eos else 0.0)


@pytest.fixture
def tiny_kenlm(tmp_path):
    arpa = tmp_path / "tiny_twi.arpa"
    arpa.write_text(
        r"""\data\
ngram 1=6
ngram 2=7
ngram 3=6

\1-grams:
-1	<unk>	0
0	<s>	-0.30103
-0.6146491	mbenyinfo	-0.30103
-0.6146491	</s>	0
-0.7659168	ɛna	-0.30103
-0.6146491	nson	-0.30103

\2-grams:
-0.4301247	<s> mbenyinfo	-0.30103
-0.4301247	mbenyinfo mbenyinfo	-0.30103
-0.20660876	ɛna </s>	0
-0.5404639	nson </s>	0
-0.4740302	<s> ɛna	-0.30103
-0.4301247	mbenyinfo nson	-0.30103
-0.3422159	nson nson	-0.47712123

\3-grams:
-0.1638568	<s> mbenyinfo mbenyinfo
-0.09113217	<s> ɛna </s>
-0.7462621	nson nson </s>
-0.1638568	mbenyinfo mbenyinfo nson
-0.13823806	mbenyinfo nson nson
-0.13375957	nson nson nson

\end\
""",
        encoding="utf-8",
    )
    return kenlm.Model(str(arpa))


def make_processor(alpha=1.0, beta=0.0, top_k=4):
    return WhisperKenLMLogitsProcessor(
        tokenizer=TinyTokenizer(),
        lm_model=WordCountLM(),
        lm_alpha=alpha,
        lm_beta=beta,
        top_k=top_k,
        lm_token_threshold=1,
    )


def test_partial_bpe_word_is_not_scored_until_next_word_starts():
    proc = make_processor()

    assert proc.completed_word_potential([1, 2, 3]).text == ""

    first_word = proc.completed_word_potential([1, 2, 3, 4])
    assert first_word.text == "mbenyinfo"
    assert first_word.word_count == 1
    assert first_word.log10_score == pytest.approx(-1.0)

    # Continuing the new partial word adds no additional LM potential.
    continued = proc.completed_word_potential([1, 2, 3, 4, 5])
    assert continued == first_word

    final = proc.completed_word_potential([1, 2, 3, 4, 5, 99])
    assert final.text == "mbenyinfo nson"
    assert final.word_count == 2
    assert final.log10_score == pytest.approx(-2.25)
    assert final.eos is True


def test_potential_deltas_telescope_and_word_bonus_occurs_once():
    proc = make_processor(alpha=1.0, beta=0.7)
    ids = [1, 2, 3, 4, 5, 99]
    previous = proc.completed_word_potential([])
    log10_deltas = []
    word_deltas = []

    for end in range(1, len(ids) + 1):
        current = proc.completed_word_potential(ids[:end])
        log10_deltas.append(current.log10_score - previous.log10_score)
        word_deltas.append(current.word_count - previous.word_count)
        previous = current

    assert sum(log10_deltas) == pytest.approx(-2.25)
    assert sum(word_deltas) == 2
    assert word_deltas == [0, 0, 0, 1, 0, 1]


def test_log10_lm_delta_is_converted_to_natural_log():
    proc = make_processor(alpha=1.0, beta=0.0, top_k=1)
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.full((1, 100), -100.0)
    scores[0, 4] = 0.0

    adjusted = proc(input_ids, scores)

    assert adjusted[0, 4].item() == pytest.approx(-math.log(10.0))


def test_real_tiny_kenlm_deltas_telescope_in_log10_units(tiny_kenlm):
    proc = WhisperKenLMLogitsProcessor(
        tokenizer=TinyTokenizer(),
        lm_model=tiny_kenlm,
        lm_alpha=1.0,
        top_k=1,
        lm_token_threshold=1,
    )
    ids = [1, 2, 3, 4, 5, 99]
    potentials = [
        proc.completed_word_potential(ids[:end]) for end in range(len(ids) + 1)
    ]
    delta_sum = sum(
        after.log10_score - before.log10_score
        for before, after in zip(potentials, potentials[1:])
    )

    expected = tiny_kenlm.score("mbenyinfo nson", bos=True, eos=True)
    assert delta_sum == pytest.approx(expected)

    scores = torch.full((1, 100), -100.0)
    scores[0, 4] = 0.0
    adjusted = proc(torch.tensor([[1, 2, 3]]), scores)
    first_word_log10 = tiny_kenlm.score("mbenyinfo", bos=True, eos=False)
    assert adjusted[0, 4].item() == pytest.approx(
        math.log(10.0) * first_word_log10
    )


def test_alpha_beta_zero_matches_acoustic_pruner_exactly():
    control = WhisperAcousticTopKLogitsProcessor(top_k=4)
    fusion_zero = make_processor(alpha=0.0, beta=0.0, top_k=4)
    input_ids = torch.tensor([[1, 2, 3], [7, 8, 6]])
    scores = torch.randn(2, 100)

    assert torch.equal(
        control(input_ids, scores),
        fusion_zero(input_ids, scores),
    )


def test_punctuation_completes_twi_word_without_stripping_letters():
    proc = make_processor()

    partial = proc.completed_word_potential([7, 8])
    punctuated = proc.completed_word_potential([7, 8, 6])

    assert partial.text == ""
    assert punctuated.text == "ɛna"
    assert punctuated.word_count == 1
    assert punctuated.eos is False

    # Starting a new word immediately after punctuation must not remove the
    # already-completed Twi word from the potential.
    after_punctuation = proc.completed_word_potential([7, 8, 6, 1])
    assert after_punctuation.text == "ɛna"


def test_cached_waxal_tokenizer_closes_previous_real_bpe_word():
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.WhisperProcessor.from_pretrained(
            "waxal-benchmarking/whisper-small-waxal-aka",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("cached WAXAL Whisper tokenizer is unavailable")

    tokenizer = processor.tokenizer
    proc = WhisperKenLMLogitsProcessor(
        tokenizer=tokenizer,
        lm_model=WordCountLM(),
        top_k=4,
        lm_token_threshold=1,
    )
    ids = tokenizer.encode("mbenyinfo nson", add_special_tokens=False)
    pieces = [tokenizer.decode([token_id]) for token_id in ids]
    next_word_index = next(i for i, piece in enumerate(pieces) if piece.startswith(" "))

    before = proc.completed_word_potential(ids[:next_word_index])
    after = proc.completed_word_potential(ids[: next_word_index + 1])
    final = proc.completed_word_potential(ids + [tokenizer.eos_token_id])

    assert before.text == ""
    assert after.text == "mbenyinfo"
    assert final.text == "mbenyinfo nson"
