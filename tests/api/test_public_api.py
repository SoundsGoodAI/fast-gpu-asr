#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the supported Python API."""

from importlib import import_module

import pytest

import fast_gpu_asr
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError

EXPECTED_IMPORTS: dict[str, tuple[str, str]] = {
    "ASR": (".asr", "ASR"),
    "Encoder": (".encoder.encoder", "Encoder"),
    "CTCGreedyDecoder": (".decoder.zipformer_decoder", "CTCGreedyDecoder"),
    "ZipformerModifiedBeamSearchDecoder": (
        ".decoder.zipformer_decoder",
        "ZipformerModifiedBeamSearchDecoder",
    ),
    "ParakeetModifiedBeamSearchDecoder": (
        ".decoder.parakeet_decoder",
        "ParakeetModifiedBeamSearchDecoder",
    ),
    "PostProcessor": (".decoder.postprocessor", "PostProcessor"),
}


def test_public_api_exports_runtime_components() -> None:
    """Expose every documented runtime component under its canonical name."""

    assert fast_gpu_asr.__all__ == list(EXPECTED_IMPORTS)


@pytest.mark.parametrize(
    ("name", "target"),
    EXPECTED_IMPORTS.items(),
    ids=EXPECTED_IMPORTS,
)
def test_public_api_resolves_canonical_runtime_class(
    name: str,
    target: tuple[str, str],
) -> None:
    """Resolve every public alias to the exact implementation object."""

    module_name, attribute_name = target

    assert getattr(fast_gpu_asr, name) is getattr(
        import_module(module_name, "fast_gpu_asr"),
        attribute_name,
    )


def test_exception_types_preserve_distinct_catch_boundaries() -> None:
    """Keep initialization and inference failures independently catchable."""

    assert issubclass(ASRInitializationError, Exception)
    assert issubclass(ASRInferenceError, Exception)
    assert not issubclass(ASRInitializationError, ASRInferenceError)
    assert not issubclass(ASRInferenceError, ASRInitializationError)
