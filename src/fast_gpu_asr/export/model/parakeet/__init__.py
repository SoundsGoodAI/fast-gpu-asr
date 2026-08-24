#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Export-oriented PyTorch components for NVIDIA Parakeet TDT.

The package contains a NeMo-compatible waveform feature extractor,
FastConformer encoder, relative-position attention, and TDT predictor and
joiner. Its operators are structured for ONNX lowering and replacement by the
native Parakeet TensorRT plugins where specialized GPU execution is required.
"""
