#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Dependency-light PyTorch model definitions used during export.

The Parakeet and Zipformer subpackages reproduce the inference-time portions
of their upstream architectures without requiring NeMo or Icefall. Exporters
load converted checkpoint weights into these modules before lowering their
feature extractors, encoders, predictors, and joiners to ONNX and TensorRT.
These definitions are not part of the runtime public API.
"""
