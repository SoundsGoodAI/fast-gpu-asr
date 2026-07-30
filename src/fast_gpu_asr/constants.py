#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Shared constants for model export and inference."""

DECODER_TYPES = (
    "transducer_modified_beam_search",
    "transducer_greedy_search",
    "ctc_greedy_search",
)
TRANSDUCER_DECODER_TYPES = DECODER_TYPES[:2]
MODEL_TYPE_ZIPFORMER = "zipformer_asr"
MODEL_TYPE_PARAKEET = "parakeet_asr"
MODEL_CONFIG_FILE = "model_config.yaml"
ONNX_OPSET_VERSION = 20
PARAKEET_ONNX_FILE = "parakeet.onnx"
PARAKEET_TENSORRT_FILE = "parakeet.trt"
TDT_DECODER_ONNX_FILE = "tdt_decoder.onnx"
TDT_DECODER_TENSORRT_FILE = "tdt_decoder.trt"
ZIPFORMER_ONNX_FILE = "zipformer.onnx"
ZIPFORMER_TENSORRT_FILE = "zipformer.trt"
ZIPFORMER_DECODER_ONNX_FILE = "decoder.onnx"
ZIPFORMER_DECODER_TENSORRT_FILE = "decoder.trt"
ZERO_LOG = -20.7233
