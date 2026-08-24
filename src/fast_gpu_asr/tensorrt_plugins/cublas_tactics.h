// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace fastgpuasr_tensorrt
{

// TensorRT reserves tactic zero while selecting an implementation. Positive
// IDs keep that sentinel distinct from the cuBLAS compute modes serialized in
// an engine.
constexpr int32_t kStrictComputeTactic = 1;
constexpr int32_t kFast16FComputeTactic = 2;
constexpr int32_t kFast16BFComputeTactic = 3;
constexpr int32_t kFastTF32ComputeTactic = 4;
constexpr std::array<int32_t, 4> kCublasComputeTactics{
    kStrictComputeTactic,
    kFast16FComputeTactic,
    kFast16BFComputeTactic,
    kFastTF32ComputeTactic,
};

constexpr bool isCublasComputeTactic(int32_t tactic) noexcept
{
    for (int32_t const candidate : kCublasComputeTactics)
    {
        if (candidate == tactic)
        {
            return true;
        }
    }
    return false;
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

inline int32_t writeCublasComputeTactics(
    int32_t* tactics, int32_t nbTactics) noexcept
{
    if (tactics == nullptr
        || nbTactics != static_cast<int32_t>(kCublasComputeTactics.size()))
    {
        return 1;
    }
    for (std::size_t index = 0; index < kCublasComputeTactics.size(); ++index)
    {
        tactics[index] = kCublasComputeTactics[index];
    }
    return 0;
}

inline int32_t setCublasComputeTactic(
    int32_t tactic, int32_t& selectedTactic) noexcept
{
    // TensorRT reserves zero for the default tactic before autotuning.
    if (tactic == 0)
    {
        tactic = kStrictComputeTactic;
    }
    if (!isCublasComputeTactic(tactic))
    {
        return 1;
    }
    selectedTactic = tactic;
    return 0;
}

} // namespace fastgpuasr_tensorrt
