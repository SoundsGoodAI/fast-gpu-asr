#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Runtime and validation tests for CTC decoding shared by both model families."""

from typing import cast

import cupy as cp
import numpy as np
import pytest

from fast_gpu_asr.constants import INT32_MAX
from fast_gpu_asr.decoder.ctc_decoder import CTCGreedyDecoder
from fast_gpu_asr.utils import ASRInferenceError

DTYPES = (
    pytest.param(np.dtype(np.float32), id="fp32"),
    pytest.param(np.dtype(np.float16), id="fp16"),
    pytest.param(cp.dtype("bfloat16"), marks=pytest.mark.sm80, id="bf16"),
)


def make_decoder(blank_id: int, blank_penalty: float) -> CTCGreedyDecoder:
    """Create a CTC decoder on the test's current CUDA device and stream.

    Parameters
    ----------
    blank_id : int
        Vocabulary ID interpreted as CTC blank.
    blank_penalty : float
        Score subtracted from the blank token before greedy selection.

    Returns
    -------
    CTCGreedyDecoder
        Decoder configured for 40 ms encoder frames on the current device.
    """

    return CTCGreedyDecoder(
        blank_id=blank_id,
        encoder_frame_shift_sec=0.04,
        blank_penalty=blank_penalty,
        device=cp.cuda.Device(),
        stream=cp.cuda.get_current_stream(),
    )


def decode_reference(
    log_probs: np.typing.NDArray[np.float32],
    lengths: list[int],
    blank_id: int,
    penalty: float,
) -> tuple[list[list[int]], list[list[float]]]:
    """Independently select and collapse CTC paths on the CPU.

    Parameters
    ----------
    log_probs : np.typing.NDArray[np.float32]
        Batched log probabilities, after input quantization but before the
        FP32 blank penalty.
    lengths : list[int]
        Valid frame counts, clamped to each utterance's available frames.
    blank_id : int
        Vocabulary ID removed as blank after collapsing repeated labels.
    penalty : float
        Score subtracted from blank before selecting the lowest-index maximum.

    Returns
    -------
    tuple[list[list[int]], list[list[float]]]
        Collapsed token IDs and timestamps in seconds.
    """

    token_ids, timestamps = [], []
    for row, length in zip(log_probs, lengths, strict=True):
        tokens, times = [], []
        previous = blank_id
        for frame, scores in enumerate(row[: max(0, length)]):
            adjusted = scores.copy()
            adjusted[blank_id] -= np.float32(penalty)
            token = int(adjusted.argmax())
            if token != blank_id and token != previous:
                tokens.append(token)
                times.append(frame * 0.04)
            previous = token
        token_ids.append(tokens)
        timestamps.append(times)
    return token_ids, timestamps


def assert_result(
    actual: tuple[list[list[int]], list[list[float]]],
    expected: tuple[list[list[int]], list[list[float]]],
) -> None:
    """Compare ragged decoder outputs without dropping empty or extra rows.

    Parameters
    ----------
    actual : tuple[list[list[int]], list[list[float]]]
        GPU decoder output: token IDs and timestamps.
    expected : tuple[list[list[int]], list[list[float]]]
        Independent CPU result with exact token IDs and matching row lengths.
        Floating-point outputs allow GPU rounding within a small tolerance.
    """

    assert isinstance(actual, tuple)
    assert actual[0] == expected[0]
    for rows, references in zip(actual[1:], expected[1:], strict=True):
        for row, reference in zip(rows, references, strict=True):
            assert len(row) == len(reference)
            np.testing.assert_allclose(row, reference, atol=2e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("blank_id", (0, 3))
def test_ctc_greedy_collapses_repeats_and_blanks(
    dtype: np.dtype, blank_id: int
) -> None:
    decoder = make_decoder(blank_id, 0.0)
    paths = cp.array([[0, 1, 1, 0, 1, 2], [2, 2, 0, 1, 2, 1]])
    paths[paths == 0] = blank_id
    log_probs = cp.log(cp.eye(4, dtype=cp.float32)[paths]).astype(dtype)
    expected = decode_reference(log_probs.astype(cp.float32).get(), [6, 4], blank_id, 0)

    result = decoder(log_probs, cp.array([6, 4], dtype=np.int32))

    assert result[0] == [[1, 1, 2], [2, 1]]
    np.testing.assert_allclose(result[-1][0], [0.04, 0.16, 0.20])
    np.testing.assert_allclose(result[-1][1], [0.0, 0.12])
    assert_result(result, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("blank_id", (0, 1))
@pytest.mark.parametrize(
    ("blank_probability", "blank_penalty", "emits_token"),
    ((0.75, 0.0, False), (0.75, 2.0, True), (0.25, 0.0, True), (0.25, -2.0, False)),
    ids=("blank-wins", "positive-penalty", "token-wins", "negative-penalty"),
)
def test_ctc_greedy_applies_blank_penalty_before_argmax(
    blank_id: int, blank_probability: float, blank_penalty: float, emits_token: bool
) -> None:
    decoder = make_decoder(blank_id, blank_penalty)
    log_probs = cp.full((1, 1, 2), np.log(1 - blank_probability), dtype=cp.float32)
    log_probs[:, :, blank_id] = np.log(blank_probability)
    expected = decode_reference(log_probs.get(), [1], blank_id, blank_penalty)

    result = decoder(log_probs, cp.array([1], dtype=np.int32))

    assert result[0] == ([[1 - blank_id]] if emits_token else [[]])
    assert result[-1] == ([[0.0]] if emits_token else [[]])
    assert_result(result, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_ctc_greedy_breaks_ties_by_lowest_token_id(dtype: np.dtype) -> None:
    decoder = make_decoder(299, 0.0)
    log_probs = cp.full((1, 4, 300), -cp.inf, dtype=dtype)
    log_probs[0, 0, [31, 256]] = np.log(0.5)
    log_probs[0, 1, [257, 299]] = np.log(0.5)
    log_probs[0, 2, 299] = 0
    log_probs[0, 3, 257] = 0
    expected = decode_reference(log_probs.astype(cp.float32).get(), [4], 299, 0)

    result = decoder(log_probs, cp.asarray([4], dtype=cp.int32))

    assert result[0] == [[31, 257, 257]]
    assert_result(result, expected)


@pytest.mark.cuda
def test_ctc_greedy_reuses_buffers_without_leaking_results() -> None:
    decoder = make_decoder(0, 0.0)
    log_probs = cp.log(
        cp.asarray(
            [
                [
                    [0.1, 0.7, 0.1, 0.1],
                    [0.01, 0.97, 0.01, 0.01],
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ]
            ]
            * 2,
            dtype=cp.float32,
        )
    )
    first = decoder(log_probs, cp.asarray([5, 3], dtype=cp.int32))
    expected_first = decode_reference(log_probs.get(), [5, 3], 0, 0)
    assert_result(first, expected_first)
    buffer_names = (
        "emitted_tokens",
        "emitted_timestamps",
        "emitted_lengths",
        "emitted_tokens_host",
        "emitted_timestamps_host",
        "emitted_lengths_host",
    )
    buffers = {name: getattr(decoder, name) for name in buffer_names}
    swapped = log_probs[:, :, [0, 2, 1, 3]]
    second = decoder(swapped, cp.asarray([5, 3], dtype=cp.int32))
    assert_result(second, decode_reference(swapped.get(), [5, 3], 0, 0))
    assert second[0] != first[0]
    for name, buffer in buffers.items():
        assert getattr(decoder, name) is buffer, name

    for data, lengths in (
        (log_probs, [-1, 99]),
        (log_probs[:1, :3], [3]),
        (cp.full((2, 4, 4), -np.log(4), dtype=cp.float32), [4, 4]),
        (log_probs[:, :0], [0, 0]),
        (log_probs, [0, 0]),
        (log_probs, [5, 3]),
    ):
        gpu_lengths = cp.asarray(lengths, dtype=cp.int32)
        result = decoder(data, gpu_lengths)
        assert_result(result, decode_reference(data.get(), lengths, 0, 0))
        np.testing.assert_array_equal(gpu_lengths.get(), lengths)
    assert_result(first, expected_first)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("buffer_name", "message"),
    [
        ("emitted_timestamps", "CTC output buffers"),
        ("emitted_lengths_host", "CTC output buffers"),
        ("emitted_timestamps_host", "CTC host output buffers"),
    ],
)
def test_ctc_greedy_rejects_missing_reusable_buffers(
    buffer_name: str, message: str
) -> None:
    decoder = make_decoder(0, 0.3)
    log_probs = cp.log(cp.array([[[0.2, 0.8], [0.8, 0.2]]], dtype=np.float32))
    lengths = cp.array([2], dtype=np.int32)
    expected = decode_reference(log_probs.get(), [2], 0, 0.3)
    assert_result(decoder(log_probs.copy(), lengths), expected)
    buffer = getattr(decoder, buffer_name)
    setattr(decoder, buffer_name, None)

    inputs = log_probs.copy()
    with pytest.raises(ASRInferenceError, match=message):
        decoder(inputs, lengths)
    if message == "CTC output buffers":
        np.testing.assert_array_equal(inputs.get(), log_probs.get())

    setattr(decoder, buffer_name, buffer)
    assert_result(decoder(log_probs.copy(), lengths), expected)


@pytest.mark.cuda
def test_ctc_greedy_returns_empty_results_for_zero_frames() -> None:
    result = make_decoder(0, 0.0)(
        cp.empty((2, 0, 4), dtype=np.float32), cp.zeros(2, dtype=np.int32)
    )

    assert result == ([[], []], [[], []])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_ctc_greedy_collapses_across_warp_and_tile_boundaries(dtype: np.dtype) -> None:
    paths = np.tile(np.arange(771) % 2 + 1, (2, 1))
    paths[0] = 1
    paths[0, 257] = 0
    probabilities = np.full((2, 771, 4), 0.1, dtype=np.float32)
    probabilities[np.arange(2)[:, None], np.arange(771), paths] = 0.7
    for frame in (256, 700):
        probabilities[0, frame] = [0.01, 0.97, 0.01, 0.01]
    log_probs = cp.asarray(np.log(probabilities), dtype=dtype)
    expected = decode_reference(log_probs.astype(cp.float32).get(), [771, 769], 0, 0)

    actual = make_decoder(0, 0.0)(log_probs, cp.asarray([771, 769], dtype=cp.int32))

    assert actual[0][0] == [1, 1]
    assert_result(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("step", (1, -1, 2), ids=("contiguous", "reversed", "stepped"))
@pytest.mark.parametrize("penalty", (-0.5, 0.0, 0.5))
def test_ctc_greedy_blank_penalty_mutates_original_view_including_padding(
    dtype: np.dtype, step: int, penalty: float
) -> None:
    decoder = make_decoder(3, penalty)
    probabilities = np.random.default_rng(17).uniform(0.1, 1, (4, 6, 4))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    base = cp.repeat(cp.asarray(np.log(probabilities), dtype=dtype), abs(step), axis=2)
    log_probs = base[::step, ::step, ::step][:2]
    lengths = [log_probs.shape[1] - 1, 0]
    penalized = base.astype(cp.float32).get()
    penalized_view = penalized[::step, ::step, ::step][:2]
    penalized_view[:, :, 3] -= np.float32(penalty)
    penalized = penalized.astype(base.dtype).astype(np.float32)
    # Public argmax sees the already-rounded in-place penalty.
    expected = decode_reference(penalized[::step, ::step, ::step][:2], lengths, 3, 0)

    result = decoder(log_probs, cp.asarray(lengths, dtype=cp.int32))

    assert_result(result, expected)
    np.testing.assert_array_equal(base.astype(cp.float32).get(), penalized)


@pytest.mark.cuda
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_ctc_greedy_handles_large_byte_offsets(axis: int, reverse: bool) -> None:
    stride = 1 << 30
    nbytes = 3 * stride + 16
    if cp.cuda.runtime.memGetInfo()[0] < nbytes + (256 << 20):
        pytest.skip("The large-offset regression needs 3 GiB of free GPU memory.")
    storage = cp.empty(nbytes, dtype=cp.uint8)
    shape, strides = [1, 1, 4], [16, 16, 4]
    shape[axis], strides[axis] = 4, stride
    log_probs = cp.ndarray(
        tuple(shape), dtype=cp.float32, memptr=storage.data, strides=tuple(strides)
    )
    scores = np.random.default_rng(17).uniform(-3, 0, shape).astype(np.float32)
    scores -= np.log(np.exp(scores).sum(axis=2, keepdims=True))
    log_probs[:] = cp.asarray(scores)
    if reverse:
        view = [slice(None)] * 3
        view[axis] = slice(None, None, -1)
        log_probs, scores = log_probs[tuple(view)], scores[tuple(view)]
    lengths = [shape[1]] * shape[0]
    expected = decode_reference(scores, lengths, 3, 0.3)

    result = make_decoder(3, 0.3)(log_probs, cp.asarray(lengths, dtype=cp.int32))

    assert_result(result, expected)
    scores[:, :, 3] -= np.float32(0.3)
    np.testing.assert_array_equal(log_probs.get(), scores)


@pytest.mark.parametrize(
    ("log_probs", "output_lengths", "message"),
    (
        pytest.param(
            np.zeros((2, 3), np.float32), np.zeros(2, np.int32), "rank-3", id="rank"
        ),
        pytest.param(
            np.zeros((0, 3, 4), np.float32),
            np.zeros(0, np.int32),
            "At least one",
            id="empty-batch",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.int32),
            np.zeros(2, np.int32),
            "log-probability dtype",
            id="integer-log-probabilities",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.float64),
            np.zeros(2, np.int32),
            "log-probability dtype",
            id="log-probability-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.float32),
            np.zeros(3, np.int32),
            "output lengths",
            id="length-count",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.float32),
            np.zeros((2, 1), np.int32),
            "output lengths",
            id="length-rank",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.float32),
            np.zeros(2, np.int64),
            "output lengths",
            id="length-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 4), np.float32),
            np.zeros(4, np.int32)[::2],
            "contiguous int32",
            id="noncontiguous-lengths",
        ),
    ),
)
def test_ctc_decoder_rejects_malformed_inputs(
    log_probs: np.typing.NDArray[np.generic],
    output_lengths: np.typing.NDArray[np.generic],
    message: str,
) -> None:
    decoder = CTCGreedyDecoder.__new__(CTCGreedyDecoder)

    with pytest.raises(ASRInferenceError, match=message):
        decoder(cast(cp.ndarray, log_probs), cast(cp.ndarray, output_lengths))


def test_ctc_decoder_rejects_int32_frame_overflow() -> None:
    log_probs = np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=np.float32),
        shape=(1, INT32_MAX + 1, 4),
        strides=(0, 0, 0),
    )
    decoder = CTCGreedyDecoder.__new__(CTCGreedyDecoder)

    with pytest.raises(ASRInferenceError, match="CTC frame count exceeds"):
        decoder(cast(cp.ndarray, log_probs), cast(cp.ndarray, np.zeros(1, np.int32)))
