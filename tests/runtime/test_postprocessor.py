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


class FakeTokenizer:
    """Provide deterministic SentencePiece behavior and decode-call tracking."""

    pieces = ("<blk>", "1", "▁2", "<unk>", "▁", "x")

    def __init__(self, model_file: str) -> None:
        """Initialize decode-call tracking for the in-memory vocabulary.

        Parameters
        ----------
        model_file : str
            Tokenizer path accepted for compatibility with SentencePiece.
        """

        self.decode_calls: list[list[list[int]]] = []

    def vocab_size(self) -> int:
        """Return the number of pieces in the deterministic vocabulary.

        Returns
        -------
        int
            Number of tokenizer pieces.
        """

        return len(self.pieces)

    def id_to_piece(self, token_ids: list[int]) -> list[str]:
        """Map token IDs to their deterministic SentencePiece surfaces.

        Parameters
        ----------
        token_ids : list[int]
            Token IDs to resolve.

        Returns
        -------
        list[str]
            Piece surface corresponding to each token ID.
        """

        return [self.pieces[token_id] for token_id in token_ids]

    def piece_to_id(self, piece: str) -> int:
        """Return the ID of one deterministic SentencePiece surface.

        Parameters
        ----------
        piece : str
            Piece surface to resolve.

        Returns
        -------
        int
            Token ID of the requested piece.
        """

        return self.pieces.index(piece)

    def decode(self, token_batches: list[list[int]]) -> list[str]:
        """Decode batches while recording calls made by the postprocessor.

        Parameters
        ----------
        token_batches : list[list[int]]
            Batched token IDs supplied to SentencePiece.

        Returns
        -------
        list[str]
            Deterministically decoded text for each token sequence.
        """

        self.decode_calls.append(token_batches)
        texts = []
        for token_ids in token_batches:
            text = "".join(self.pieces[token_id] for token_id in token_ids)
            text = text.replace("▁", " ").replace("<unk>", " <unk> ")
            texts.append(" ".join(text.split()))
        return texts


@pytest.fixture
def postprocessor(monkeypatch: pytest.MonkeyPatch) -> PostProcessor:
    """Construct a postprocessor with the deterministic test tokenizer.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the SentencePiece processor constructor.

    Returns
    -------
    PostProcessor
        Postprocessor backed by ``FakeTokenizer`` at 1 kHz.
    """

    monkeypatch.setattr(
        postprocessor_module.spm, "SentencePieceProcessor", FakeTokenizer
    )
    return PostProcessor(Path("bpe.model"), sample_rate=1000)


@pytest.fixture(scope="module")
def real_postprocessor(tmp_path_factory: pytest.TempPathFactory) -> PostProcessor:
    """Build a minimal real SentencePiece-backed postprocessor.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to create the module-scoped tokenizer workspace.

    Returns
    -------
    PostProcessor
        Postprocessor backed by a trained byte-fallback SentencePiece model.
    """

    model_dir = tmp_path_factory.mktemp("postprocessor-sentencepiece")
    corpus_path = model_dir / "corpus.txt"
    corpus_path.write_text("é b hello world\nhello b é world\n", encoding="utf-8")
    model_prefix = model_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=300,
        byte_fallback=True,
        control_symbols=["<ctrl>"],
        hard_vocab_limit=False,
        unk_surface="⁇",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    return PostProcessor(model_prefix.with_suffix(".model"), sample_rate=1000)


def postprocess_one(
    postprocessor: PostProcessor,
    token_ids: list[int],
    timestamps: list[float],
) -> tuple[str, list[tuple[str, float, float]]]:
    """Postprocess one 100 ms utterance.

    Parameters
    ----------
    postprocessor : PostProcessor
        Postprocessor under test.
    token_ids : list[int]
        Decoder token IDs for the utterance.
    timestamps : list[float]
        Token start times in seconds.

    Returns
    -------
    tuple[str, list[tuple[str, float, float]]]
        Decoded text and word-level ``(word, start, end)`` tuples.
    """

    texts, word_timestamps = postprocessor(
        [np.zeros(100, dtype=np.float32)],
        [token_ids],
        [timestamps],
    )
    return texts[0], word_timestamps[0]


def test_postprocessor_groups_sentencepiece_words(
    postprocessor: PostProcessor,
) -> None:
    assert postprocess_one(postprocessor, [1, 5, 2], [0.0, 0.02, 0.04]) == (
        "1x 2",
        [("1x", 0.0, 0.04), ("2", 0.04, 0.1)],
    )
    assert postprocessor.tokenizer.decode_calls == [[[1, 5, 2]]]


def test_postprocessor_handles_standalone_boundaries(
    postprocessor: PostProcessor,
) -> None:
    audio = np.zeros(100, dtype=np.float32)

    assert postprocessor(
        [audio, audio, audio],
        [[4, 4, 2, 4], [1, 4, 5], [4, 4]],
        [[0.01, 0.02, 0.04, 0.08], [0.0, 0.03, 0.04], [0.01, 0.02]],
    ) == (
        ["2", "1 x", ""],
        [
            [("2", 0.01, 0.1)],
            [("1", 0.0, 0.03), ("x", 0.03, 0.1)],
            [],
        ],
    )


def test_postprocessor_batches_fallback_words(postprocessor: PostProcessor) -> None:
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
    assert postprocessor.tokenizer.decode_calls == [
        [[1, 5, 2], [1, 3, 2, 5]],
        [[1, 3], [2, 5]],
    ]


def test_postprocessor_handles_empty_inputs(
    postprocessor: PostProcessor,
) -> None:
    audio = np.zeros(100, dtype=np.float32)
    texts, word_timestamps = postprocessor(
        [audio, audio, audio], [[], [1], []], [[], [0.01], []]
    )
    assert texts == ["", "1", ""]
    assert word_timestamps == [[], [("1", 0.01, 0.1)], []]

    word_timestamps[0].append(("word", 0.0, 0.1))
    assert word_timestamps[2] == []

    postprocessor.tokenizer.decode = lambda _: pytest.fail(
        "An empty batch should not be sent to SentencePiece."
    )

    assert postprocessor([], [], []) == ([], [])


@pytest.mark.parametrize(
    ("pieces", "timestamps", "expected"),
    (
        pytest.param(
            ("▁", "<0xC3>", "<0xA9>", "▁", "b"),
            [0.01, 0.02, 0.02, 0.04, 0.05],
            ("é b", [("é", 0.01, 0.04), ("b", 0.04, 0.1)]),
            id="byte-fallback",
        ),
        pytest.param(
            ("▁", "<ctrl>", "▁", "b"),
            [0.01, 0.02, 0.04, 0.05],
            ("b", [("b", 0.04, 0.1)]),
            id="control-only-word",
        ),
        pytest.param(
            ("<unk>", "▁", "b"),
            [0.0, 0.04, 0.05],
            ("⁇ b", [("⁇", 0.0, 0.04), ("b", 0.04, 0.1)]),
            id="unknown-surface",
        ),
    ),
)
def test_postprocessor_handles_real_sentencepiece_special_tokens(
    real_postprocessor: PostProcessor,
    pieces: tuple[str, ...],
    timestamps: list[float],
    expected: tuple[str, list[tuple[str, float, float]]],
) -> None:
    token_ids = [real_postprocessor.tokenizer.piece_to_id(piece) for piece in pieces]

    assert postprocess_one(real_postprocessor, token_ids, timestamps) == expected


def test_postprocessor_rejects_unaligned_batches(
    postprocessor: PostProcessor,
) -> None:
    audio = np.zeros(100, dtype=np.float32)
    inputs = (([], [[]], [[]]), ([audio], [[]], []))

    for audios, token_ids, timestamps in inputs:
        with pytest.raises(ASRInferenceError, match="batch size differs"):
            postprocessor(audios, token_ids, timestamps)


def test_postprocessor_rejects_unaligned_token_timestamps(
    postprocessor: PostProcessor,
) -> None:
    with pytest.raises(ASRInferenceError, match="counts differ"):
        postprocessor(
            [np.zeros(100, dtype=np.float32)],
            [[1, 1]],
            [[0.0]],
        )


def test_postprocessor_rejects_invalid_timestamps(
    postprocessor: PostProcessor,
) -> None:
    for timestamps in ([-0.01], [float("nan")], [0.04, 0.02]):
        with pytest.raises(
            ASRInferenceError, match="finite, non-negative, nondecreasing"
        ):
            postprocess_one(postprocessor, [1] * len(timestamps), timestamps)


def test_postprocessor_rejects_malformed_audio(
    postprocessor: PostProcessor,
) -> None:
    for audio in (
        np.zeros(0, dtype=np.float32),
        np.zeros((1, 100), dtype=np.float32),
    ):
        with pytest.raises(ASRInferenceError, match="nonempty one-dimensional"):
            postprocessor([audio], [[]], [[]])


def test_postprocessor_clamps_float32_endpoint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm, "SentencePieceProcessor", FakeTokenizer
    )
    postprocessor = PostProcessor(Path("bpe.model"), sample_rate=16_000)
    audio = np.zeros(1_656, dtype=np.float32)

    assert postprocessor(
        [audio], [[1]], [[float(np.float32(audio.size / 16_000))]]
    ) == (["1"], [[("1", 0.103, 0.103)]])


def test_postprocessor_rounds_and_clamps_word_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocessor_module.spm, "SentencePieceProcessor", FakeTokenizer
    )
    postprocessor = PostProcessor(Path("bpe.model"), sample_rate=10_000)

    assert postprocessor(
        [
            np.zeros(1_006, dtype=np.float32),
            np.zeros(1_000, dtype=np.float32),
            np.zeros(1_000, dtype=np.float32),
        ],
        [[1, 2], [1, 2], [1, 4, 2]],
        [[0.0016, 0.0236], [0.0, 0.2], [0.0, 0.2, 0.2]],
    ) == (
        ["1 2", "1 2", "1 2"],
        [
            [("1", 0.002, 0.024), ("2", 0.024, 0.101)],
            [("1", 0.0, 0.1), ("2", 0.1, 0.1)],
            [("1", 0.0, 0.1), ("2", 0.1, 0.1)],
        ],
    )
