// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <NvInferRuntimeBase.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <span>

namespace fastgpuasr_tensorrt
{

// TensorRT reserves tactic zero while selecting an implementation. Positive
// IDs keep that sentinel distinct from the cuBLAS compute modes serialized in
// an engine.
inline constexpr int32_t kStrictComputeTactic = 1;
inline constexpr int32_t kFast16FComputeTactic = 2;
inline constexpr int32_t kFast16BFComputeTactic = 3;
inline constexpr int32_t kFastTF32ComputeTactic = 4;
inline constexpr std::array<int32_t, 4> kCublasComputeTactics{
    kStrictComputeTactic,
    kFast16FComputeTactic,
    kFast16BFComputeTactic,
    kFastTF32ComputeTactic,
};

// The FAST_* modes require FP32 A, B, and C storage and alter the internal
// multiplication precision. GEMMs with FP16 or BF16 storage therefore retain
// strict FP32 accumulation.
inline constexpr std::array<int32_t, 1> kReducedStorageCublasComputeTactics{
    kStrictComputeTactic,
};

// Query only during tactic enumeration, never on the inference hot path.
inline bool deviceSupportsAmpereCompute() noexcept
{
    int device{};
    int major{};
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)
            == cudaSuccess
        && major >= 8;
}

constexpr std::span<int32_t const> getCublasComputeTactics(
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT,
    bool ampereCompute = true) noexcept
{
    switch (type)
    {
    case nvinfer1::DataType::kBF16:
        if (!ampereCompute)
        {
            return {};
        }
        [[fallthrough]];
    case nvinfer1::DataType::kHALF:
        return kReducedStorageCublasComputeTactics;
    case nvinfer1::DataType::kFLOAT:
        return std::span<int32_t const>(kCublasComputeTactics).first(ampereCompute ? 4 : 2);
    default: return {};
    }
}

constexpr bool isCublasComputeTactic(int32_t tactic,
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT) noexcept
{
    for (int32_t const candidate : getCublasComputeTactics(type))
    {
        if (candidate == tactic)
        {
            return true;
        }
    }
    return false;
}

constexpr int32_t getCublasComputeTacticCount(
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT,
    bool ampereCompute = true) noexcept
{
    return static_cast<int32_t>(getCublasComputeTactics(type, ampereCompute).size());
}

constexpr cublasComputeType_t getCublasComputeType(int32_t tactic) noexcept
{
    switch (tactic)
    {
    case kStrictComputeTactic: return CUBLAS_COMPUTE_32F;
    case kFast16FComputeTactic: return CUBLAS_COMPUTE_32F_FAST_16F;
    case kFast16BFComputeTactic: return CUBLAS_COMPUTE_32F_FAST_16BF;
    case kFastTF32ComputeTactic: return CUBLAS_COMPUTE_32F_FAST_TF32;
    // Callers validate tactics before execution. Retain strict FP32 as a safe
    // fallback if this helper is nevertheless called with an invalid value.
    default: return CUBLAS_COMPUTE_32F;
    }
}

inline int32_t writeCublasComputeTactics(int32_t* tactics,
    int32_t nbTactics,
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT,
    bool ampereCompute = true) noexcept
{
    std::span<int32_t const> const validTactics =
        getCublasComputeTactics(type, ampereCompute);
    if (tactics == nullptr
        || validTactics.empty()
        || nbTactics != static_cast<int32_t>(validTactics.size()))
    {
        return 1;
    }
    for (int32_t const tactic : validTactics)
    {
        *tactics++ = tactic;
    }
    return 0;
}

inline int32_t setCublasComputeTactic(int32_t tactic,
    int32_t& selectedTactic,
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT) noexcept
{
    // TensorRT restores a serialized tactic before runtime shapes are known.
    // The default FP32 type deliberately accepts every compute tactic here;
    // onShapeChange validates it again against the concrete tensor type.
    if (tactic == 0)
    {
        tactic = kStrictComputeTactic;
    }
    if (!isCublasComputeTactic(tactic, type))
    {
        return 1;
    }
    selectedTactic = tactic;
    return 0;
}

} // namespace fastgpuasr_tensorrt
