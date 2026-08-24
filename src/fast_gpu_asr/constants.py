#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Shared constants for model export and inference."""

import torch

# Model and decoder configuration.
DECODER_TYPES = (
    "transducer_modified_beam_search",
    "transducer_greedy_search",
    "ctc_greedy_search",
)
MODEL_TYPE_PARAKEET = "parakeet_asr"
MODEL_TYPE_ZIPFORMER = "zipformer_asr"
PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
TRANSDUCER_DECODER_TYPES = DECODER_TYPES[:2]

# Model bundle artifacts.
MODEL_CONFIG_FILE = "model_config.yaml"
PARAKEET_DECODER_ONNX_FILE = "tdt_decoder.onnx"
PARAKEET_DECODER_TENSORRT_FILE = "tdt_decoder.trt"
PARAKEET_ONNX_FILE = "parakeet.onnx"
PARAKEET_TENSORRT_FILE = "parakeet.trt"
ZIPFORMER_DECODER_CONTEXTS_FILE = "decoder_contexts.pt"
ZIPFORMER_DECODER_ONNX_FILE = "decoder.onnx"
ZIPFORMER_DECODER_TENSORRT_FILE = "decoder.trt"
ZIPFORMER_ONNX_FILE = "zipformer.onnx"
ZIPFORMER_TENSORRT_FILE = "zipformer.trt"

# ONNX and TensorRT plugin identifiers.
ONNX_OPSET_VERSION = 20
PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME = "parakeet_conformer_convolution"
PARAKEET_FEATURE_PLUGIN_NAME = "parakeet_feature_extractor"
PARAKEET_FLASH_ATTENTION_PLUGIN_NAME = "parakeet_flash_attention"
TENSORRT_PLUGIN_NAMESPACE = "fast_gpu_asr"
ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME = "zipformer_attention_value"
ZIPFORMER_CONVOLUTION_PLUGIN_NAME = "zipformer_convolution"
ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME = "zipformer_downsample"
ZIPFORMER_FEATURE_PLUGIN_NAME = "zipformer_feature_extractor"
ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME = "zipformer_output_assembly"
ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME = "zipformer_relative_attention"
ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME = "zipformer_upsample_bypass"

# Runtime limits and tuning values.
# Amortize thread scheduling over roughly 64 MB of float32 waveform data.
AUDIO_SAMPLES_PER_WORKER = 16_000_000
INT32_MAX = (1 << 31) - 1
# Keep this even so each graph replay restores canonical ping-pong buffer roles.
TDT_SEARCH_CHUNK_STEPS = 8
ZIPFORMER_BEAM_SEARCH_THREADS = 512
ZERO_LOG = -20.7233
