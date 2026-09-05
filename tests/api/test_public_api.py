#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the supported Python API."""

from importlib import import_module

import fast_gpu_asr

EXPECTED_IMPORTS = {
    "ASR": ".asr",
    "Encoder": ".encoder.encoder",
    "CTCGreedyDecoder": ".decoder.zipformer_decoder",
    "ZipformerModifiedBeamSearchDecoder": ".decoder.zipformer_decoder",
    "ParakeetModifiedBeamSearchDecoder": ".decoder.parakeet_decoder",
    "PostProcessor": ".decoder.postprocessor",
}


def test_public_api_exports_canonical_runtime_classes() -> None:
    assert set(fast_gpu_asr.__all__) == set(EXPECTED_IMPORTS)
    assert len(fast_gpu_asr.__all__) == len(EXPECTED_IMPORTS)
    for name, module_name in EXPECTED_IMPORTS.items():
        expected = getattr(import_module(module_name, "fast_gpu_asr"), name)
        assert isinstance(expected, type), name
        assert getattr(fast_gpu_asr, name) is expected, name
