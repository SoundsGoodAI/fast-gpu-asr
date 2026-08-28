// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace fastgpuasr_tensorrt
{

// Keep this value synchronized with TENSORRT_PLUGIN_NAMESPACE in
// fast_gpu_asr/constants.py. TensorRT uses it for ONNX lookup, creator
// registration, and engine deserialization, so a mismatch makes bundles
// unloadable.
constexpr char kPluginNamespace[] = "fast_gpu_asr";

} // namespace fastgpuasr_tensorrt
