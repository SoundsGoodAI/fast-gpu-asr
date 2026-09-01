#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for SentencePiece text and word-timestamp postprocessing."""

from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm

import fast_gpu_asr.decoder.postprocessor as postprocessor_module
from fast_gpu_asr.decoder.postprocessor import PostProcessor
from fast_gpu_asr.utils import ASRInferenceError

REAL_SENTENCEPIECE_PROCESSOR = spm.SentencePieceProcessor


class FakeTokenizer:
    """Expose the SentencePiece behavior needed by postprocessor tests."""

    def __init__(self, model_file: str) -> None:
        self.model_file = model_file

    def piece_to_id(self, piece: str) -> int:
        return {"<blk>": 0, "▁": 4}[piece]

    def vocab_size(self) -> int:
        return 6

    def id_to_piece(self, token_id: int | list[int]) -> str | list[str]:
        pieces = {0: "<blk>", 1: "1", 2: "▁2", 3: "<unk>", 4: "▁", 5: "x"}
        if isinstance(token_id, list):
            return [pieces[item] for item in token_id]
        return pieces[token_id]

    def decode(
        self,
        tokens: list[int] | list[str] | list[list[int]],
    ) -> str | list[str]:
        if tokens and isinstance(tokens[0], list):
            return [self.decode(token_ids) for token_ids in tokens]  # type: ignore[misc]
        pieces = (
            [self.id_to_piece(token) for token in tokens]  # type: ignore[arg-type]
            if not tokens or isinstance(tokens[0], int)
            else tokens
        )
        return "".join(pieces).replace("▁", " ").strip()  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def patch_sentencepiece(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace SentencePiece with a deterministic in-memory vocabulary."""

    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        FakeTokenizer,
    )


@pytest.fixture(scope="module")
def real_tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny tokenizer with byte, control, and user-defined pieces."""

    model_dir = tmp_path_factory.mktemp("postprocessor-sentencepiece")
    corpus_path = model_dir / "corpus.txt"
    corpus_path.write_text(
        "é b hello world user symbols\nhello b é world\n",
        encoding="utf-8",
    )
    model_prefix = model_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=300,
        model_type="unigram",
        byte_fallback=True,
        control_symbols=["<ctrl>", "CTRL"],
        user_defined_symbols=["<user>"],
        hard_vocab_limit=False,
        unk_surface="⁇",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    return model_prefix.with_suffix(".model")


def make_postprocessor() -> PostProcessor:
    """Construct a postprocessor with the deterministic test tokenizer."""

    return PostProcessor(Path("bpe.model"), sample_rate=1000)


def postprocess_one(
    token_ids: list[int],
    timestamps: list[float],
    utterance_end_sec: float,
) -> tuple[str, list[tuple[str, float, float]]]:
    """Postprocess one synthetic decoder result."""

    postprocessor = make_postprocessor()
    texts, word_timestamps = postprocessor(
        [np.zeros(round(utterance_end_sec * 1000), dtype=np.float32)],
        [token_ids],
        [timestamps],
    )
    return texts[0], word_timestamps[0]


def test_postprocessor_normalizes_sentencepiece_boundaries() -> None:
    text, word_timestamps = postprocess_one(
        [4, 2, 4],
        [0.04, 0.08, 0.12],
        0.2,
    )

    assert text == "2"
    assert word_timestamps == [("2", 0.04, 0.2)]


def test_postprocessor_groups_subword_pieces() -> None:
    text, word_timestamps = postprocess_one(
        [1, 5, 2],
        [0.0, 0.02, 0.04],
        0.1,
    )

    assert text == "1x 2"
    assert word_timestamps == [("1x", 0.0, 0.04), ("2", 0.04, 0.1)]


def test_postprocessor_reuses_batched_text_for_ordinary_words() -> None:
    postprocessor = make_postprocessor()
    decode_calls: list[list[list[int]]] = []

    def track_decode(
        tokens: list[int] | list[str] | list[list[int]],
    ) -> str | list[str]:
        assert tokens == [[1, 5, 2]]
        decode_calls.append(tokens)
        return ["1x 2"]

    postprocessor.tokenizer.decode = track_decode

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [[1, 5, 2]],
        [[0.0, 0.02, 0.04]],
    ) == (["1x 2"], [[("1x", 0.0, 0.04), ("2", 0.04, 0.1)]])
    assert decode_calls == [[[1, 5, 2]]]


def test_postprocessor_batches_exact_fallback_for_unusual_whitespace() -> None:
    postprocessor = make_postprocessor()
    decode_calls: list[list[list[int]]] = []

    def decode(
        tokens: list[int] | list[str] | list[list[int]],
    ) -> str | list[str]:
        assert tokens and isinstance(tokens[0], list)
        decode_calls.append(tokens)
        if tokens == [[1, 5, 2], [1, 3, 2, 5]]:
            return ["1x 2", "1 <unk> 2x"]
        if tokens == [[1, 3], [2, 5]]:
            return ["1 <unk>", "2x"]
        pytest.fail(f"Unexpected tokenizer input: {tokens}")

    postprocessor.tokenizer.decode = decode

    assert postprocessor(
        [np.zeros(100, dtype=np.float32), np.zeros(200, dtype=np.float32)],
        [[1, 5, 2], [1, 3, 2, 5]],
        [[0.0, 0.02, 0.04], [0.0, 0.02, 0.04, 0.06]],
    ) == (
        ["1x 2", "1 <unk> 2x"],
        [
            [("1x", 0.0, 0.04), ("2", 0.04, 0.1)],
            [("1 <unk>", 0.0, 0.04), ("2x", 0.04, 0.2)],
        ],
    )
    assert decode_calls == [
        [[1, 5, 2], [1, 3, 2, 5]],
        [[1, 3], [2, 5]],
    ]


def test_postprocessor_preserves_special_pieces() -> None:
    text, word_timestamps = postprocess_one(
        [3, 2],
        [0.0, 0.04],
        0.1,
    )

    assert text == "<unk> 2"
    assert word_timestamps == [("<unk>", 0.0, 0.04), ("2", 0.04, 0.1)]


def test_postprocessor_handles_empty_token_sequence() -> None:
    postprocessor = make_postprocessor()

    texts, word_timestamps = postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [[]],
        [[]],
    )

    assert texts == [""]
    assert word_timestamps == [[]]


def test_postprocessor_handles_empty_batch() -> None:
    postprocessor = make_postprocessor()
    postprocessor.tokenizer.decode = lambda _: pytest.fail(
        "An empty batch should not be sent to SentencePiece."
    )

    assert postprocessor([], [], []) == ([], [])


def test_postprocessor_does_not_treat_unknown_as_missing_standalone_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TokenizerWithoutStandalone(FakeTokenizer):
        def vocab_size(self) -> int:
            return 4

        def piece_to_id(self, piece: str) -> int:
            return {"<blk>": 0, "▁": 3}[piece]

    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        TokenizerWithoutStandalone,
    )
    postprocessor = PostProcessor(Path("bpe.model"), sample_rate=1000)

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [[3, 2]],
        [[0.0, 0.04]],
    ) == (["<unk> 2"], [[("<unk>", 0.0, 0.04), ("2", 0.04, 0.1)]])


def test_postprocessor_retains_earliest_consecutive_standalone_boundary() -> None:
    text, word_timestamps = postprocess_one(
        [4, 4, 2],
        [0.01, 0.02, 0.04],
        0.1,
    )

    assert text == "2"
    assert word_timestamps == [("2", 0.01, 0.1)]


def test_postprocessor_decodes_byte_fallback_into_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    real_tokenizer_path: Path,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        REAL_SENTENCEPIECE_PROCESSOR,
    )
    postprocessor = PostProcessor(real_tokenizer_path, sample_rate=1000)
    tokenizer = postprocessor.tokenizer
    ids = [
        tokenizer.piece_to_id("▁"),
        tokenizer.piece_to_id("<0xC3>"),
        tokenizer.piece_to_id("<0xA9>"),
        tokenizer.piece_to_id("▁"),
        tokenizer.piece_to_id("b"),
    ]

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [ids],
        [[0.01, 0.02, 0.02, 0.04, 0.05]],
    ) == (["é b"], [[("é", 0.01, 0.04), ("b", 0.04, 0.1)]])


@pytest.mark.parametrize("control_piece", ("<ctrl>", "CTRL"))
def test_postprocessor_ignores_control_tokens_without_consuming_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    real_tokenizer_path: Path,
    control_piece: str,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        REAL_SENTENCEPIECE_PROCESSOR,
    )
    postprocessor = PostProcessor(real_tokenizer_path, sample_rate=1000)
    tokenizer = postprocessor.tokenizer
    ids = [
        tokenizer.piece_to_id("▁"),
        tokenizer.piece_to_id(control_piece),
        tokenizer.piece_to_id("b"),
    ]

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [ids],
        [[0.01, 0.02, 0.03]],
    ) == (["b"], [[("b", 0.01, 0.1)]])


@pytest.mark.parametrize("control_piece", ("<ctrl>", "CTRL"))
def test_postprocessor_control_preserves_byte_run_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    real_tokenizer_path: Path,
    control_piece: str,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        REAL_SENTENCEPIECE_PROCESSOR,
    )
    postprocessor = PostProcessor(real_tokenizer_path, sample_rate=1000)
    tokenizer = postprocessor.tokenizer
    ids = [
        tokenizer.piece_to_id("▁"),
        tokenizer.piece_to_id("<0xC3>"),
        tokenizer.piece_to_id(control_piece),
        tokenizer.piece_to_id("<0xA9>"),
    ]

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [ids],
        [[0.01, 0.02, 0.03, 0.04]],
    ) == (["��"], [[("��", 0.01, 0.1)]])


def test_postprocessor_uses_sentencepiece_unknown_surface(
    monkeypatch: pytest.MonkeyPatch,
    real_tokenizer_path: Path,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm,
        "SentencePieceProcessor",
        REAL_SENTENCEPIECE_PROCESSOR,
    )
    postprocessor = PostProcessor(real_tokenizer_path, sample_rate=1000)
    tokenizer = postprocessor.tokenizer
    ids = [
        tokenizer.unk_id(),
        tokenizer.piece_to_id("▁"),
        tokenizer.piece_to_id("b"),
    ]

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [ids],
        [[0.0, 0.04, 0.05]],
    ) == (["⁇ b"], [[("⁇", 0.0, 0.04), ("b", 0.04, 0.1)]])


def test_postprocessor_rejects_unaligned_batch_dimensions() -> None:
    postprocessor = make_postprocessor()

    with pytest.raises(ASRInferenceError, match="batch size differs"):
        postprocessor(
            [np.zeros(100, dtype=np.float32)],
            [[], []],
            [[]],
        )


def test_postprocessor_rejects_unaligned_token_timestamps() -> None:
    postprocessor = make_postprocessor()

    with pytest.raises(ASRInferenceError, match="counts differ"):
        postprocessor(
            [np.zeros(100, dtype=np.float32)],
            [[1, 2]],
            [[0.0]],
        )


@pytest.mark.parametrize(
    "timestamps",
    (
        [-0.01],
        [float("nan")],
        [float("inf")],
        [0.04, 0.02],
    ),
    ids=("negative", "nan", "infinite", "decreasing"),
)
def test_postprocessor_rejects_invalid_timestamps(timestamps: list[float]) -> None:
    postprocessor = make_postprocessor()
    token_ids = [1] * len(timestamps)

    with pytest.raises(
        ASRInferenceError,
        match="finite, non-negative, nondecreasing",
    ):
        postprocessor(
            [np.zeros(100, dtype=np.float32)],
            [token_ids],
            [timestamps],
        )


@pytest.mark.parametrize(
    "audio",
    (
        np.zeros(0, dtype=np.float32),
        np.zeros((1, 100), dtype=np.float32),
    ),
)
def test_postprocessor_rejects_malformed_audio(audio: object) -> None:
    postprocessor = make_postprocessor()

    with pytest.raises(ASRInferenceError, match="nonempty one-dimensional"):
        postprocessor(
            [audio],  # type: ignore[list-item]
            [[]],
            [[]],
        )


def test_postprocessor_accepts_float32_timestamp_at_audio_endpoint() -> None:
    postprocessor = make_postprocessor()

    assert postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [[1]],
        [[float(np.float32(0.1))]],
    ) == (["1"], [[("1", 0.1, 0.1)]])


def test_postprocessor_clamps_tolerated_float32_endpoint_drift() -> None:
    postprocessor = PostProcessor(Path("bpe.model"), sample_rate=16_000)
    audio = np.zeros(1_656, dtype=np.float32)

    texts, word_timestamps = postprocessor(
        [audio],
        [[1]],
        [[float(np.float32(len(audio) / 16_000))]],
    )

    assert texts == ["1"]
    assert word_timestamps == [[("1", 0.103, 0.103)]]


def test_postprocessor_clamps_standalone_boundary_to_audio_endpoint() -> None:
    _, word_timestamps = postprocess_one(
        [1, 4, 2],
        [0.0, 0.2, 0.2],
        0.1,
    )

    assert word_timestamps == [("1", 0.0, 0.1), ("2", 0.1, 0.1)]
