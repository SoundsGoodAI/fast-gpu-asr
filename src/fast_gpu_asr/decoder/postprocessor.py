#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Transcription postprocessing."""

from math import isfinite
from pathlib import Path

import sentencepiece as spm

from ..utils import ASRInferenceError


class PostProcessor:
    """Convert decoder token sequences into text and word timestamps."""

    def __init__(self, tokenizer_path: Path) -> None:
        """Cache tokenizer metadata used during transcription postprocessing.

        Unknown, control, and byte-fallback tokens require exact per-word decoding
        because their surfaces can disagree with the token word boundaries.

        Parameters
        ----------
        tokenizer_path : Path
            Path to the SentencePiece model packaged with the ASR model.
        """

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.starts_word = tuple(
            self.tokenizer.id_to_piece(token_id).startswith("▁")
            for token_id in range(self.tokenizer.vocab_size())
        )
        self.standalone_word_id = self.tokenizer.piece_to_id("▁")
        if self.tokenizer.is_unknown(self.standalone_word_id):
            self.standalone_word_id = None
        self.fallback_token_ids = {
            token_id
            for token_id in range(self.tokenizer.vocab_size())
            if self.tokenizer.is_unknown(token_id)
            or self.tokenizer.is_control(token_id)
            or self.tokenizer.is_byte(token_id)
        }

    def __call__(
        self,
        audio_durations: list[float],
        token_ids: list[list[int]],
        timestamps: list[list[float]],
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Convert decoded tokens into text and word intervals.

        Parameters
        ----------
        audio_durations : list[float]
            Audio durations in seconds, used to bound word timestamps.
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
            Raised when batch dimensions or token metadata counts differ or
            a decoder timestamp is malformed.
        """

        if not len(audio_durations) == len(token_ids) == len(timestamps):
            raise ASRInferenceError(
                "Decoder batch size differs from the input audio batch size."
            )

        if not token_ids:
            return [], []

        texts = self.tokenizer.decode(token_ids)
        word_timestamps: list[list[tuple[str, float, float]]] = [[] for _ in token_ids]
        fallback_tokens: list[list[int]] = []
        fallback_words: list[tuple[int, float, float]] = []
        rounded_timestamps: dict[float, float] = {}
        for utt_idx, (utt_end_sec, utt_token_ids, utt_timestamps) in enumerate(
            zip(audio_durations, token_ids, timestamps, strict=True)
        ):
            if len(utt_token_ids) != len(utt_timestamps):
                raise ASRInferenceError(
                    "Decoder token and timestamp counts differ for utterance "
                    f"{utt_idx}: {len(utt_token_ids)} tokens and {len(utt_timestamps)} "
                    "timestamps."
                )

            if not utt_token_ids:
                continue

            word_boundaries: list[tuple[int, float, float]] = []
            word_left: int | None = None
            rounded_word_start = 0.0
            pending_word_start: float | None = None
            previous_timestamp = 0.0
            for token_index, (token_id, timestamp) in enumerate(
                zip(utt_token_ids, utt_timestamps, strict=True)
            ):
                if not isfinite(timestamp) or timestamp < previous_timestamp:
                    raise ASRInferenceError(
                        "Expected finite, non-negative, nondecreasing decoder "
                        f"timestamps, got {timestamp} at token {token_index} of "
                        f"utterance {utt_idx}."
                    )

                previous_timestamp = timestamp

                if token_id == self.standalone_word_id:
                    # Consecutive standalone markers retain the earliest boundary.
                    if pending_word_start is None:
                        pending_word_start = min(timestamp, utt_end_sec)
                    continue

                if pending_word_start is not None:
                    next_word_start = pending_word_start
                    pending_word_start = None
                elif word_left is None or self.starts_word[token_id]:
                    next_word_start = min(timestamp, utt_end_sec)
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
                rounded_word_end = rounded_timestamps.get(utt_end_sec)
                if rounded_word_end is None:
                    rounded_word_end = round(utt_end_sec, 3)
                    rounded_timestamps[utt_end_sec] = rounded_word_end

                word_boundaries.append(
                    (word_left, rounded_word_start, rounded_word_end)
                )

            # Special tokens can hide mismatches even when word counts agree.
            words = texts[utt_idx].split()
            can_split_words = self.fallback_token_ids.isdisjoint(utt_token_ids)
            if can_split_words and len(words) == len(word_boundaries):
                word_timestamps[utt_idx] = [
                    (word, word_start, word_end)
                    for word, (_, word_start, word_end) in zip(
                        words, word_boundaries, strict=True
                    )
                ]
                continue

            word_rights = [boundary[0] for boundary in word_boundaries[1:]]
            word_rights.append(len(utt_token_ids))
            for (left, word_start, word_end), right in zip(
                word_boundaries, word_rights, strict=True
            ):
                fallback_tokens.append(
                    [
                        token_id
                        for token_id in utt_token_ids[left:right]
                        if token_id != self.standalone_word_id
                    ]
                )
                fallback_words.append((utt_idx, word_start, word_end))

        if fallback_tokens:
            decoded_words = self.tokenizer.decode(fallback_tokens)
            for word, (utt_idx, word_start, word_end) in zip(
                decoded_words, fallback_words, strict=True
            ):
                word = word.strip()
                if word:
                    word_timestamps[utt_idx].append((word, word_start, word_end))

        return texts, word_timestamps
