#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Transcription postprocessing."""

from pathlib import Path

import numpy as np
import sentencepiece as spm

from ..utils import ASRInferenceError


class PostProcessor:
    """Convert decoder token sequences into text and word timestamps."""

    def __init__(self, tokenizer_path: Path, sample_rate: int) -> None:
        """Cache tokenizer metadata used during transcription postprocessing.

        Parameters
        ----------
        tokenizer_path : Path
            Path to the SentencePiece model packaged with the ASR model.
        sample_rate : int
            Input audio sampling rate in hertz.
        """

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.sample_rate = sample_rate

        token_pieces = self.tokenizer.id_to_piece(
            list(range(self.tokenizer.vocab_size()))
        )
        self.starts_word = tuple(piece.startswith("▁") for piece in token_pieces)
        self.standalone_word_id = self.tokenizer.piece_to_id("▁")

    def __call__(
        self,
        audios: list[np.typing.NDArray[np.float32]],
        token_ids: list[list[int]],
        timestamps: list[list[float]],
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Convert decoded tokens into text and word intervals.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Input waveforms used to bound the final word in each utterance.
        token_ids : list[list[int]]
            Decoded token IDs for each utterance.
        timestamps : list[list[float]]
            Corresponding token start timestamps in seconds.

        Returns
        -------
        tuple[list[str], list[list[tuple[str, float, float]]]]
            Decoded texts and ``(word, start, end)`` tuples in input order.

        Raises
        ------
        ASRInferenceError
            Raised when batch dimensions or token and timestamp counts differ or
            an audio waveform is malformed.
        """

        if not len(audios) == len(token_ids) == len(timestamps):
            raise ASRInferenceError(
                "Decoder batch size differs from the input audio batch size."
            )

        texts = self.tokenizer.decode(token_ids)
        word_timestamps: list[list[tuple[str, float, float]]] = [[] for _ in token_ids]
        fallback_token_ids: list[list[int]] = []
        fallback_words: list[tuple[int, float, float]] = []
        rounded_timestamps: dict[float, float] = {}
        for utt_idx, (audio, utt_token_ids, utt_timestamps) in enumerate(
            zip(audios, token_ids, timestamps, strict=True)
        ):
            if audio.ndim != 1 or audio.size == 0:
                raise ASRInferenceError(
                    "Expected each audio waveform to be a nonempty one-dimensional "
                    f"NumPy array, got utterance {utt_idx}."
                )
            if len(utt_token_ids) != len(utt_timestamps):
                raise ASRInferenceError(
                    "Decoder token and timestamp counts differ for utterance "
                    f"{utt_idx}: {len(utt_token_ids)} tokens and {len(utt_timestamps)} "
                    "timestamps."
                )

            if not utt_token_ids:
                continue

            utterance_end_sec = audio.size / self.sample_rate
            word_boundaries: list[tuple[int, float, float]] = []
            word_left: int | None = None
            rounded_word_start = 0.0
            pending_word_start: float | None = None
            for token_index, (token_id, timestamp) in enumerate(
                zip(utt_token_ids, utt_timestamps, strict=True)
            ):
                if token_id == self.standalone_word_id:
                    # Consecutive standalone markers retain the earliest boundary.
                    if pending_word_start is None:
                        pending_word_start = min(timestamp, utterance_end_sec)
                    continue

                if pending_word_start is not None:
                    next_word_start = pending_word_start
                    pending_word_start = None
                elif word_left is None or self.starts_word[token_id]:
                    next_word_start = min(timestamp, utterance_end_sec)
                else:
                    continue

                rounded_next_word_start = rounded_timestamps.get(next_word_start)
                if rounded_next_word_start is None:
                    rounded_next_word_start = round(next_word_start, 3)
                    rounded_timestamps[next_word_start] = rounded_next_word_start

                if word_left is not None:
                    word_boundaries.append(
                        (word_left, rounded_word_start, rounded_next_word_start)
                    )

                word_left = token_index
                rounded_word_start = rounded_next_word_start

            if word_left is not None:
                rounded_word_end = rounded_timestamps.get(utterance_end_sec)
                if rounded_word_end is None:
                    rounded_word_end = round(utterance_end_sec, 3)
                    rounded_timestamps[utterance_end_sec] = rounded_word_end

                word_boundaries.append(
                    (word_left, rounded_word_start, rounded_word_end)
                )

            # SentencePiece normally separates exactly the boundaries identified
            # by its metaspace tokens. Reusing the decoded text avoids one native
            # decoder call per word. Unusual surfaces such as <unk> may introduce
            # additional whitespace and take the exact batched fallback below.
            words = texts[utt_idx].split()
            if len(words) == len(word_boundaries):
                word_timestamps[utt_idx] = [
                    (word, word_start, word_end)
                    for word, (_, word_start, word_end) in zip(
                        words, word_boundaries, strict=True
                    )
                ]
                continue

            word_rights = [word_left for word_left, _, _ in word_boundaries[1:]]
            word_rights.append(len(utt_token_ids))
            for (left, word_start, word_end), right in zip(
                word_boundaries, word_rights, strict=True
            ):
                fallback_token_ids.append(
                    [
                        token_id
                        for token_id in utt_token_ids[left:right]
                        if token_id != self.standalone_word_id
                    ]
                )
                fallback_words.append((utt_idx, word_start, word_end))

        if fallback_token_ids:
            decoded_words = self.tokenizer.decode(fallback_token_ids)
            for word, (utt_idx, word_start, word_end) in zip(
                decoded_words, fallback_words, strict=True
            ):
                word = word.strip()
                if word:
                    word_timestamps[utt_idx].append((word, word_start, word_end))

        return texts, word_timestamps
