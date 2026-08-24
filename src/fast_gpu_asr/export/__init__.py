#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Export Zipformer and Parakeet checkpoints as Fast GPU ASR bundles.

The command-line exporters validate upstream configuration, reconstruct the
required PyTorch inference graph, write ONNX intermediates, build TensorRT
engines for the target GPU, and package runtime configuration and tokenizer
assets. TensorRT engines are hardware- and TensorRT-version-specific and must
be rebuilt for incompatible deployment environments.
"""
