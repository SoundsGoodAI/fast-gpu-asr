#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT execution for fixed-capacity offline ASR encoders.

The encoder runtime batches mono waveforms into reusable pinned and device
buffers, executes a packaged Zipformer or Parakeet engine on a shared CuPy
stream, and returns encoder outputs and valid frame counts on the GPU. Stable
input shapes are captured in CUDA graphs for subsequent replay.
"""
