#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Decoder and public transcription result containers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecoderResult:
    """Decoded token IDs and token start timestamps for each utterance."""

    token_ids: list[list[int]]
    timestamps: list[list[float]]


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """One decoded word and its half-open time interval in seconds."""

    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Transcription:
    """Decoded text and word-level timestamps for one input waveform."""

    text: str
    words: list[WordTimestamp]
