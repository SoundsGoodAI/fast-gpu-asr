#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""High-throughput TensorRT inference for offline speech recognition.

The public API exposes the complete runtime pipeline as well as its encoder,
decoder, and postprocessing components. Export and TensorRT-plugin modules
remain internal implementation details.
"""

from .asr import ASR
from .decoder.parakeet_decoder import ParakeetModifiedBeamSearchDecoder
from .decoder.postprocessor import PostProcessor
from .decoder.zipformer_decoder import (
    CTCGreedyDecoder,
    ZipformerModifiedBeamSearchDecoder,
)
from .encoder.encoder import Encoder

__all__ = [
    "ASR",
    "Encoder",
    "CTCGreedyDecoder",
    "ZipformerModifiedBeamSearchDecoder",
    "ParakeetModifiedBeamSearchDecoder",
    "PostProcessor",
]
