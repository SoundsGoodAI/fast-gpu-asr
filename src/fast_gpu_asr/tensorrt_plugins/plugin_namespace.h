// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace fastgpuasr_tensorrt
{

// Keep this value synchronized with TENSORRT_PLUGIN_NAMESPACE in constants.py.
// TensorRT uses it for both ONNX plugin lookup and creator registration, so a
// mismatch prevents the ONNX graph from being parsed.
constexpr char kPluginNamespace[] = "fast_gpu_asr";

} // namespace fastgpuasr_tensorrt
