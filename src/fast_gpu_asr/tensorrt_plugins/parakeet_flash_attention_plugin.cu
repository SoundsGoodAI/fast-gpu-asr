// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <string_view>

#include "cublas_tactics.h"
#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

// TensorRT loads every plugin library with global symbol visibility. Keep all
// implementation details local to this shared object so identically named
// helpers in another plugin cannot be interposed by the dynamic linker.
namespace
{

// Inputs use the projection-friendly contiguous layouts emitted by ONNX:
//   qkv            [N, T, 3 * C]
//   position       [1, 2 * T - 1, C]
//   content_bias   [H, D]
//   position_bias  [H, D]
//   valid_lengths  [N]
// where C == H * D. One preparation kernel adds both learned query biases while
// converting Q to the head-major layout required by the score GEMMs. K, V, and
// relative positions remain in their original projection buffers and are read
// with strided leading dimensions. The output is contiguous [N, T, C].
// Relative alignment and masked softmax are fused in one warp-row CUDA kernel.
// Unlike online FlashAttention, this implementation materializes both score
// matrices in TensorRT-owned workspace; the plugin name refers to the fused
// relative-attention execution path.
constexpr char const* kPluginName = "parakeet_flash_attention";
constexpr char const* kPluginVersion = "1";
constexpr char const* kScaleField = "scale";
constexpr char const* kTimingCacheId = "layout=nt3c-ntc;relative=2t-1;queries=fused;softmax=warp4";
constexpr size_t kTimingCacheIdSize = 128;
constexpr int32_t kInputCount = 5;
constexpr int32_t kOutputCount = 1;
constexpr int32_t kMaximumSequenceLength = 512;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 4;
constexpr int32_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int32_t kPointerArrayCount = 3;
constexpr size_t kCublasWorkspaceBytes = 16U << 20;
constexpr float kLog2E = 1.4426950408889634F;

struct WorkspaceLayout
{
    size_t contentQueryOffset{};
    size_t positionQueryOffset{};
    size_t positionScoresOffset{};
    size_t attentionWeightsOffset{};
    size_t pointerArraysOffset{};
    size_t cublasWorkspaceOffset{};
    size_t totalBytes{};
};

constexpr bool isSupportedDataType(DataType type) noexcept
{
    return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

constexpr int32_t dataTypeBytes(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kFLOAT: return sizeof(float);
    case DataType::kHALF: return sizeof(half);
    case DataType::kBF16: return sizeof(__nv_bfloat16);
    default: return 0;
    }
}

bool checkedMultiply(size_t left, size_t right, size_t& output) noexcept
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    {
        return false;
    }
    output = left * right;
    return true;
}

bool checkedAdd(size_t left, size_t right, size_t& output) noexcept
{
    if (right > std::numeric_limits<size_t>::max() - left)
    {
        return false;
    }
    output = left + right;
    return true;
}

bool checkedAlignUp(size_t value, size_t alignment, size_t& output) noexcept
{
    size_t adjusted{};
    if (alignment == 0 || !checkedAdd(value, alignment - 1, adjusted))
    {
        return false;
    }
    output = adjusted / alignment * alignment;
    return true;
}

bool hasAddressableBytes(Dims const& dims, int32_t elementBytes) noexcept
{
    if (dims.nbDims < 1 || elementBytes < 1)
    {
        return false;
    }
    size_t elements = 1;
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (dims.d[index] < 1
            || !checkedMultiply(elements, static_cast<size_t>(dims.d[index]), elements))
        {
            return false;
        }
    }
    size_t bytes{};
    return checkedMultiply(elements, static_cast<size_t>(elementBytes), bytes)
           && bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max());
}

bool sameShape(Dims const& left, Dims const& right) noexcept
{
    if (left.nbDims != right.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < left.nbDims; ++index)
    {
        if (left.d[index] != right.d[index])
        {
            return false;
        }
    }
    return true;
}

bool computeWorkspaceLayout(Dims const& qkv, Dims const& position, Dims const& bias, DataType type,
    WorkspaceLayout& layout) noexcept
{
    int32_t const elementBytes = dataTypeBytes(type);
    if (qkv.nbDims != 3 || position.nbDims != 3 || bias.nbDims != 2 || elementBytes == 0)
    {
        return false;
    }
    for (int32_t index = 0; index < qkv.nbDims; ++index)
    {
        if (qkv.d[index] < 1 || position.d[index] < 1)
        {
            return false;
        }
    }
    if (bias.d[0] < 1 || bias.d[1] < 1)
    {
        return false;
    }

    size_t batchHeads{};
    size_t queryElements{};
    size_t positionElements{};
    size_t attentionElements{};
    size_t pointerBytes{};
    size_t queryBytes{};
    size_t positionBytes{};
    size_t attentionBytes{};
    size_t offset{};
    if (!checkedMultiply(static_cast<size_t>(qkv.d[0]), static_cast<size_t>(bias.d[0]), batchHeads)
        || !checkedMultiply(
            static_cast<size_t>(qkv.d[0]), static_cast<size_t>(qkv.d[1]), queryElements)
        || !checkedMultiply(queryElements, static_cast<size_t>(position.d[2]), queryElements)
        || !checkedMultiply(queryElements, static_cast<size_t>(elementBytes), queryBytes)
        || !checkedMultiply(batchHeads, static_cast<size_t>(qkv.d[1]), positionElements)
        || !checkedMultiply(positionElements, static_cast<size_t>(position.d[1]), positionElements)
        || !checkedMultiply(positionElements, static_cast<size_t>(elementBytes), positionBytes)
        || !checkedMultiply(batchHeads, static_cast<size_t>(qkv.d[1]), attentionElements)
        || !checkedMultiply(attentionElements, static_cast<size_t>(qkv.d[1]), attentionElements)
        || !checkedMultiply(attentionElements, static_cast<size_t>(elementBytes), attentionBytes)
        || !checkedMultiply(batchHeads, kPointerArrayCount * sizeof(void*), pointerBytes))
    {
        return false;
    }

    layout.contentQueryOffset = 0;
    if (!checkedAlignUp(queryBytes, 256, offset))
    {
        return false;
    }
    layout.positionQueryOffset = offset;
    if (!checkedAdd(offset, queryBytes, offset) || !checkedAlignUp(offset, 256, offset))
    {
        return false;
    }
    layout.positionScoresOffset = offset;
    // The two quadratic score matrices are reused across all plugin instances
    // by TensorRT's workspace planner. Pointer arrays follow them at 256-byte
    // boundaries, and the final region is handed directly to cuBLAS.
    if (!checkedAdd(offset, positionBytes, offset) || !checkedAlignUp(offset, 256, offset))
    {
        return false;
    }
    layout.attentionWeightsOffset = offset;
    if (!checkedAdd(offset, attentionBytes, offset) || !checkedAlignUp(offset, 256, offset))
    {
        return false;
    }
    layout.pointerArraysOffset = offset;
    if (!checkedAdd(offset, pointerBytes, offset) || !checkedAlignUp(offset, 256, offset))
    {
        return false;
    }
    layout.cublasWorkspaceOffset = offset;
    if (!checkedAdd(offset, kCublasWorkspaceBytes, layout.totalBytes))
    {
        return false;
    }
    return layout.totalBytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max());
}

bool haveValidShapes(Dims const* inputs, Dims const& output, DataType type) noexcept
{
    if (inputs == nullptr || inputs[0].nbDims != 3 || inputs[1].nbDims != 3 || inputs[2].nbDims != 2
        || inputs[3].nbDims != 2 || inputs[4].nbDims != 1 || output.nbDims != 3
        || !isSupportedDataType(type))
    {
        return false;
    }

    int32_t const elementBytes = dataTypeBytes(type);
    for (int32_t index = 0; index < 4; ++index)
    {
        if (!hasAddressableBytes(inputs[index], elementBytes))
        {
            return false;
        }
    }
    if (!hasAddressableBytes(inputs[4], sizeof(int32_t))
        || !hasAddressableBytes(output, elementBytes))
    {
        return false;
    }

    int64_t const batch = inputs[0].d[0];
    int64_t const length = inputs[0].d[1];
    int64_t const channels = inputs[1].d[2];
    int64_t const heads = inputs[2].d[0];
    int64_t const headDim = inputs[2].d[1];
    // cuBLAS leading dimensions are int32_t. K and V advance through the
    // fused projection with a 3C stride, so C itself must fit after scaling.
    if (length > kMaximumSequenceLength || channels < 1
        || channels > std::numeric_limits<int32_t>::max() / 3)
    {
        return false;
    }
    int64_t const relativeLength = 2 * length - 1;
    if (inputs[0].d[2] != 3 * channels || heads * headDim != channels
        || batch > std::numeric_limits<int32_t>::max() / heads || inputs[1].d[0] != 1
        || inputs[1].d[1] != relativeLength || !sameShape(inputs[2], inputs[3])
        || inputs[4].d[0] != batch || output.d[0] != batch || output.d[1] != length
        || output.d[2] != channels)
    {
        return false;
    }

    int64_t const batchHeads = batch * heads;
    if (batchHeads > std::numeric_limits<int32_t>::max() / length)
    {
        return false;
    }

    WorkspaceLayout layout{};
    return computeWorkspaceLayout(inputs[0], inputs[1], inputs[2], type, layout);
}

template <typename T> __device__ __forceinline__ T addQueryBias(T query, T bias)
{
    return query + bias;
}

template <> __device__ __forceinline__ half addQueryBias<half>(half query, half bias)
{
    return __float2half(__half2float(query) + __half2float(bias));
}

template <>
__device__ __forceinline__ __nv_bfloat16 addQueryBias<__nv_bfloat16>(
    __nv_bfloat16 query, __nv_bfloat16 bias)
{
    return __float2bfloat16(__bfloat162float(query) + __bfloat162float(bias));
}

template <typename T>
__global__ void prepareQueriesAndPositionPointers(T const* qkv, T const* position,
    T const* contentBias, T const* positionBias, T* contentQuery, T* positionQuery,
    T* positionScores, void const** positionPointers, void const** queryPointers,
    void** outputPointers, int32_t batchSize, int32_t numHeads, int32_t sequenceLength,
    int32_t relativeLength, int32_t channels, int32_t headDim)
{
    int64_t const pointerCount = static_cast<int64_t>(batchSize) * numHeads;
    int64_t const queryCount = static_cast<int64_t>(batchSize) * sequenceLength * channels;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t const first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t index = first; index < pointerCount; index += stride)
    {
        int32_t const head = static_cast<int32_t>(index % numHeads);
        positionPointers[index] = position + static_cast<int64_t>(head) * headDim;
        queryPointers[index] = positionQuery + index * sequenceLength * headDim;
        outputPointers[index] = positionScores + index * sequenceLength * relativeLength;
    }
    for (int64_t index = first; index < queryCount; index += stride)
    {
        int64_t const batchStride = static_cast<int64_t>(sequenceLength) * channels;
        int32_t const batch = static_cast<int32_t>(index / batchStride);
        int64_t const indexInBatch = index - batch * batchStride;
        int32_t const frame = static_cast<int32_t>(indexInBatch / channels);
        int32_t const channel = static_cast<int32_t>(indexInBatch - frame * channels);
        int32_t const head = channel / headDim;
        int32_t const headChannel = channel - head * headDim;
        int64_t const queryOffset =
            (static_cast<int64_t>(batch) * sequenceLength + frame) * 3 * channels + channel;
        int64_t const outputOffset =
            ((static_cast<int64_t>(batch) * numHeads + head) * sequenceLength + frame) * headDim
            + headChannel;
        T const query = qkv[queryOffset];
        contentQuery[outputOffset] = addQueryBias(query, contentBias[channel]);
        positionQuery[outputOffset] = addQueryBias(query, positionBias[channel]);
    }
}

__global__ void initializeContentPointers(void const* qkv, void const* contentQuery,
    void* contentScores, void const** keyPointers, void const** queryPointers,
    void** outputPointers, int32_t batchSize, int32_t numHeads, int32_t sequenceLength,
    int32_t channels, int32_t headDim, int32_t elementBytes)
{
    int64_t const count = static_cast<int64_t>(batchSize) * numHeads;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t const first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t index = first; index < count; index += stride)
    {
        int32_t const batch = static_cast<int32_t>(index / numHeads);
        int32_t const head = static_cast<int32_t>(index % numHeads);
        auto const* qkvBytes = static_cast<unsigned char const*>(qkv);
        auto const* queryBytes = static_cast<unsigned char const*>(contentQuery);
        auto* outputBytes = static_cast<unsigned char*>(contentScores);
        int64_t const keyOffset = ((static_cast<int64_t>(batch) * sequenceLength * 3 + 1) * channels
                                      + static_cast<int64_t>(head) * headDim)
                                  * elementBytes;
        keyPointers[index] = qkvBytes + keyOffset;
        queryPointers[index] = queryBytes + index * sequenceLength * headDim * elementBytes;
        outputPointers[index] =
            outputBytes + index * sequenceLength * sequenceLength * elementBytes;
    }
}

__global__ void initializeValuePointers(void const* qkv, void const* attentionWeights, void* output,
    void const** valuePointers, void const** attentionPointers, void** outputPointers,
    int32_t batchSize, int32_t numHeads, int32_t sequenceLength, int32_t channels, int32_t headDim,
    int32_t elementBytes)
{
    // Keep the loop index wider than the validated pointer count for the final
    // grid-stride increment, matching the query preparation kernel.
    int64_t const count = static_cast<int64_t>(batchSize) * numHeads;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t const first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t index = first; index < count; index += stride)
    {
        int32_t const batch = static_cast<int32_t>(index / numHeads);
        int32_t const head = static_cast<int32_t>(index % numHeads);
        int64_t const valueOffset =
            ((static_cast<int64_t>(batch) * sequenceLength * 3 + 2) * channels
                + static_cast<int64_t>(head) * headDim)
            * elementBytes;
        int64_t const outputOffset = (static_cast<int64_t>(batch) * sequenceLength * channels
                                         + static_cast<int64_t>(head) * headDim)
                                     * elementBytes;
        auto const* valueBytes = static_cast<unsigned char const*>(qkv);
        auto const* attentionBytes = static_cast<unsigned char const*>(attentionWeights);
        auto* outputBytes = static_cast<unsigned char*>(output);
        valuePointers[index] = valueBytes + valueOffset;
        attentionPointers[index] =
            attentionBytes
            + static_cast<int64_t>(index) * sequenceLength * sequenceLength * elementBytes;
        outputPointers[index] = outputBytes + outputOffset;
    }
}

__device__ __forceinline__ float warpMaximum(float value)
{
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return __shfl_sync(0xFFFFFFFFU, value, 0);
}

__device__ __forceinline__ float warpSum(float value)
{
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return __shfl_sync(0xFFFFFFFFU, value, 0);
}

template <typename T> __device__ __forceinline__ float scalarToFloat(T value);

template <> __device__ __forceinline__ float scalarToFloat<float>(float value) { return value; }

template <> __device__ __forceinline__ float scalarToFloat<half>(half value)
{
    return __half2float(value);
}

template <> __device__ __forceinline__ float scalarToFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T floatToScalar(float value);

template <> __device__ __forceinline__ float floatToScalar<float>(float value) { return value; }

template <> __device__ __forceinline__ half floatToScalar<half>(float value)
{
    return __float2half(value);
}

template <> __device__ __forceinline__ __nv_bfloat16 floatToScalar<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename T, int MaxSlots>
__global__ void parakeetRelativeSoftmax(T const* positionScores, int32_t const* validLengths,
    T const* contentScores, T* attentionWeights, int32_t batchSize, int32_t numHeads,
    int32_t sequenceLength, int32_t relativeLength, float scaleLog2)
{
    int32_t const lane = threadIdx.x & (kWarpSize - 1);
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const row = blockIdx.x * kWarpsPerBlock + warp;
    int32_t const rowsPerBatch = numHeads * sequenceLength;
    int32_t const numRows = batchSize * rowsPerBatch;
    if (row >= numRows)
    {
        return;
    }

    int32_t const batch = row / rowsPerBatch;
    int32_t const rowInBatch = row - batch * rowsPerBatch;
    int32_t const head = rowInBatch / sequenceLength;
    int32_t const query = rowInBatch - head * sequenceLength;
    int32_t const validLength = validLengths[batch];
    int64_t const outputBase = static_cast<int64_t>(row) * sequenceLength;
    int64_t const positionBase = static_cast<int64_t>(row) * relativeLength;
    int32_t const numSlots = (sequenceLength + kWarpSize - 1) / kWarpSize;

    // One warp owns one query row. Every lane retains up to 16 keys in
    // registers, covering the production limit of 512 frames without shared
    // memory or an intermediate relative-shift tensor. Matching NeMo, valid
    // lengths mask keys only. A nonpositive length has no valid softmax domain,
    // so return an all-zero row. Lengths at or above T leave the row unmasked.
    // Query rows remain defined because downstream lengths determine which
    // frames are consumed.
    if (validLength <= 0)
    {
        for (int32_t key = lane; key < sequenceLength; key += kWarpSize)
        {
            attentionWeights[outputBase + key] = floatToScalar<T>(0.0F);
        }
        return;
    }

    float values[MaxSlots];
    float localMaximum = -FLT_MAX;
#pragma unroll
    for (int32_t slot = 0; slot < MaxSlots; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        // A finite valid score must always outrank padding, even when its value
        // is below the historical -1000 sentinel. -FLT_MAX also keeps the
        // subsequent exp2f subtraction well-defined for every nonempty row.
        float value = -FLT_MAX;
        if (slot < numSlots && key < sequenceLength && key < validLength)
        {
            // Transformer-XL relative shift: row q consumes position
            // T - 1 - q + k. Applying it here avoids a T x (2T - 1) shuffle.
            int32_t const relative = sequenceLength - 1 - query + key;
            value = (scalarToFloat(contentScores[outputBase + key])
                        + scalarToFloat(positionScores[positionBase + relative]))
                    * scaleLog2;
        }
        values[slot] = value;
        if (slot < numSlots && key < sequenceLength)
        {
            localMaximum = fmaxf(localMaximum, value);
        }
    }

    float const maximum = warpMaximum(localMaximum);
    float localSum = 0.0F;
#pragma unroll
    for (int32_t slot = 0; slot < MaxSlots; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        float const value =
            slot < numSlots && key < sequenceLength ? exp2f(values[slot] - maximum) : 0.0F;
        values[slot] = value;
        localSum += value;
    }

    float const inverseDenominator = 1.0F / warpSum(localSum);
#pragma unroll
    for (int32_t slot = 0; slot < MaxSlots; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        if (slot < numSlots && key < sequenceLength)
        {
            attentionWeights[outputBase + key] =
                floatToScalar<T>(values[slot] * inverseDenominator);
        }
    }
}

#define LAUNCH_SOFTMAX_CASE(SLOTS, TYPE)                                                           \
    case SLOTS:                                                                                    \
        parakeetRelativeSoftmax<TYPE, SLOTS><<<blocks, kThreadsPerBlock, 0, stream>>>(             \
            static_cast<TYPE const*>(positionScores), validLengths,                                \
            static_cast<TYPE const*>(contentScores), static_cast<TYPE*>(attentionWeights),         \
            batchSize, numHeads, sequenceLength, relativeLength, scaleLog2);                       \
        break

template <typename T>
bool launchParakeetSoftmax(void const* positionScores, int32_t const* validLengths,
    void const* contentScores, void* attentionWeights, int32_t batchSize, int32_t numHeads,
    int32_t sequenceLength, int32_t relativeLength, float scale, cudaStream_t stream)
{
    int32_t const rows = batchSize * numHeads * sequenceLength;
    // Avoid overflowing the validated int32 row count during ceiling division.
    uint32_t const blocks = static_cast<uint32_t>((rows - 1) / kWarpsPerBlock + 1);
    float const scaleLog2 = scale * kLog2E;
    switch ((sequenceLength + kWarpSize - 1) / kWarpSize)
    {
        LAUNCH_SOFTMAX_CASE(1, T);
        LAUNCH_SOFTMAX_CASE(2, T);
        LAUNCH_SOFTMAX_CASE(3, T);
        LAUNCH_SOFTMAX_CASE(4, T);
        LAUNCH_SOFTMAX_CASE(5, T);
        LAUNCH_SOFTMAX_CASE(6, T);
        LAUNCH_SOFTMAX_CASE(7, T);
        LAUNCH_SOFTMAX_CASE(8, T);
        LAUNCH_SOFTMAX_CASE(9, T);
        LAUNCH_SOFTMAX_CASE(10, T);
        LAUNCH_SOFTMAX_CASE(11, T);
        LAUNCH_SOFTMAX_CASE(12, T);
        LAUNCH_SOFTMAX_CASE(13, T);
        LAUNCH_SOFTMAX_CASE(14, T);
        LAUNCH_SOFTMAX_CASE(15, T);
        LAUNCH_SOFTMAX_CASE(16, T);
    default: return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

#undef LAUNCH_SOFTMAX_CASE

struct ParakeetFlashAttentionParameters
{
    float scale{};
};

class ParakeetFlashAttentionPlugin final : public IPluginV3,
                                           public IPluginV3OneCore,
                                           public IPluginV3OneBuild,
                                           public IPluginV3OneRuntime
{
  public:
    explicit ParakeetFlashAttentionPlugin(
        ParakeetFlashAttentionParameters parameters, int32_t tactic = kStrictComputeTactic) noexcept
        : mParameters(parameters), mTactic(tactic)
    {
        int const timingCacheLength = std::snprintf(mTimingCacheId.data(), mTimingCacheId.size(),
            "%s;scale=%a", kTimingCacheId, static_cast<double>(parameters.scale));
        bool const validTimingCacheId =
            timingCacheLength > 0 && static_cast<size_t>(timingCacheLength) < mTimingCacheId.size();
        mInitialized = validTimingCacheId && cublasCreate(&mCublas) == CUBLAS_STATUS_SUCCESS;
        initializeFields();
    }

    ~ParakeetFlashAttentionPlugin() override
    {
        if (mCublas != nullptr)
        {
            cublasDestroy(mCublas);
        }
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        default: return nullptr;
        }
    }

    IPluginV3* clone() noexcept override
    {
        auto* plugin = new (std::nothrow) ParakeetFlashAttentionPlugin(mParameters, mTactic);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        plugin->mInputType = mInputType;
        return plugin;
    }

    char const* getPluginName() const noexcept override { return kPluginName; }

    char const* getPluginVersion() const noexcept override { return kPluginVersion; }

    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    int32_t getNbOutputs() const noexcept override { return kOutputCount; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        // Symbolic dimensions can have independently conservative profile
        // bounds. Validate invariant metadata and addressability here; exact
        // cross-input shape relationships are checked by onShapeChange().
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !mInitialized || inputs[0].desc.dims.nbDims != 3
            || inputs[1].desc.dims.nbDims != 3 || inputs[2].desc.dims.nbDims != 2
            || inputs[3].desc.dims.nbDims != 2 || inputs[4].desc.dims.nbDims != 1
            || outputs[0].desc.dims.nbDims != 3 || !isSupportedDataType(inputs[0].desc.type)
            || inputs[4].desc.type != DataType::kINT32
            || outputs[0].desc.type != inputs[0].desc.type
            || inputs[4].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index = 0; index < 4; ++index)
        {
            if (inputs[index].desc.type != inputs[0].desc.type
                || inputs[index].desc.format != TensorFormat::kLINEAR
                || !hasAddressableBytes(inputs[index].max, dataTypeBytes(inputs[0].desc.type)))
            {
                return 1;
            }
        }
        if (!hasAddressableBytes(inputs[4].max, sizeof(int32_t))
            || !hasAddressableBytes(outputs[0].max, dataTypeBytes(outputs[0].desc.type))
            || inputs[0].max.d[1] > kMaximumSequenceLength)
        {
            return 1;
        }
        mInputType = inputs[0].desc.type;
        return 0;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !isSupportedDataType(inputTypes[0])
            || inputTypes[4] != DataType::kINT32)
        {
            return 1;
        }
        for (int32_t index = 1; index < 4; ++index)
        {
            if (inputTypes[index] != inputTypes[0])
            {
                return 1;
            }
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        static_cast<void>(shapeInputs);
        static_cast<void>(exprBuilder);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || nbShapeInputs != 0 || inputs[0].nbDims != 3
            || inputs[1].nbDims != 3 || inputs[2].nbDims != 2 || inputs[3].nbDims != 2
            || inputs[4].nbDims != 1 || inputs[0].d[0] == nullptr || inputs[0].d[1] == nullptr
            || inputs[1].d[2] == nullptr)
        {
            return 1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = inputs[1].d[2];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputCount || nbOutputs != kOutputCount || pos < 0
            || pos >= kInputCount + kOutputCount)
        {
            return false;
        }
        auto const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 4)
        {
            return desc.type == DataType::kINT32;
        }
        return isSupportedDataType(desc.type) && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount)
        {
            return 0;
        }
        WorkspaceLayout layout{};
        return computeWorkspaceLayout(
                   inputs[0].max, inputs[1].max, inputs[2].max, inputs[0].desc.type, layout)
                   ? layout.totalBytes
                   : 0;
    }

    int32_t getNbTactics() noexcept override
    {
        return getCublasComputeTacticCount(mInputType, deviceSupportsAmpereCompute());
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        return writeCublasComputeTactics(
            tactics, nbTactics, mInputType, deviceSupportsAmpereCompute());
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        return setCublasComputeTactic(tactic, mTactic, mInputType);
    }

    char const* getTimingCacheID() noexcept override
    {
        // TensorRT adds concrete shapes and formats to this implementation
        // identity. The exact scale suffix completes the creation-state key.
        return mTimingCacheId.data();
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !mInitialized || !isSupportedDataType(inputs[0].type)
            || inputs[4].type != DataType::kINT32 || outputs[0].type != inputs[0].type
            || inputs[4].format != TensorFormat::kLINEAR
            || outputs[0].format != TensorFormat::kLINEAR
            || !isCublasComputeTactic(mTactic, inputs[0].type))
        {
            return 1;
        }
        for (int32_t index = 0; index < 4; ++index)
        {
            if (inputs[index].type != inputs[0].type
                || inputs[index].format != TensorFormat::kLINEAR)
            {
                return 1;
            }
        }
        Dims inputShapes[kInputCount]{};
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            inputShapes[index] = inputs[index].dims;
        }
        return haveValidShapes(inputShapes, outputs[0].dims, inputs[0].type) ? 0 : 1;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || outputs[0] == nullptr || workspace == nullptr)
        {
            return 1;
        }
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            if (inputs[index] == nullptr)
            {
                return 1;
            }
        }

        // TensorRT calls onShapeChange first. Rechecking exact runtime shapes
        // here keeps all pointer arithmetic safe for nonstandard invocations.
        if (onShapeChange(inputDesc, kInputCount, outputDesc, kOutputCount) != 0)
        {
            return 1;
        }

        // Preserve errors from earlier asynchronous work instead of silently
        // attributing success to this invocation.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        // haveValidShapes() has proved that these 64-bit TensorRT dimensions
        // and every product consumed by int32 CUDA/cuBLAS APIs are in range.
        int32_t const batch = static_cast<int32_t>(inputDesc[0].dims.d[0]);
        int32_t const length = static_cast<int32_t>(inputDesc[0].dims.d[1]);
        int32_t const channels = static_cast<int32_t>(inputDesc[1].dims.d[2]);
        int32_t const heads = static_cast<int32_t>(inputDesc[2].dims.d[0]);
        int32_t const headDim = static_cast<int32_t>(inputDesc[2].dims.d[1]);
        int32_t const relativeLength = static_cast<int32_t>(inputDesc[1].dims.d[1]);
        DataType const inputType = inputDesc[0].type;
        int32_t const elementBytes = dataTypeBytes(inputType);
        cudaDataType_t const dataType = inputType == DataType::kHALF   ? CUDA_R_16F
                                        : inputType == DataType::kBF16 ? CUDA_R_16BF
                                                                       : CUDA_R_32F;
        int32_t const pointerCount = batch * heads;

        WorkspaceLayout layout{};
        if (!computeWorkspaceLayout(
                inputDesc[0].dims, inputDesc[1].dims, inputDesc[2].dims, inputType, layout))
        {
            return 1;
        }

        auto* workspaceBytes = static_cast<unsigned char*>(workspace);
        void* contentQuery = workspaceBytes + layout.contentQueryOffset;
        void* positionQuery = workspaceBytes + layout.positionQueryOffset;
        void* positionScores = workspaceBytes + layout.positionScoresOffset;
        auto* attentionWeights = workspaceBytes + layout.attentionWeightsOffset;
        auto* pointerArrays = workspaceBytes + layout.pointerArraysOffset;
        auto pointerArray = [pointerArrays, pointerCount](int32_t index)
        {
            return pointerArrays
                   + static_cast<size_t>(index) * static_cast<size_t>(pointerCount) * sizeof(void*);
        };
        auto** positionPointers = reinterpret_cast<void const**>(pointerArray(0));
        auto** queryPointers = reinterpret_cast<void const**>(pointerArray(1));
        auto** outputPointers = reinterpret_cast<void**>(pointerArray(2));
        auto* cublasWorkspace = workspaceBytes + layout.cublasWorkspaceOffset;

        // All three products use pointer-batched GEMMs because the shared
        // relative-position tensor and interleaved NTC QKV storage do not have
        // one uniform stride across every (batch, head) matrix. Reusing these
        // three arrays keeps workspace small and was faster than retaining nine
        // arrays initialized in one larger kernel.

        bool const streamChanged = stream != mStream;
        if (streamChanged)
        {
            if (cublasSetStream(mCublas, stream) != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mStream = stream;
            // cublasSetStream resets the handle workspace.
            mWorkspace = nullptr;
        }
        if (cublasWorkspace != mWorkspace
            && cublasSetWorkspace(mCublas, cublasWorkspace, kCublasWorkspaceBytes)
                   != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }
        mWorkspace = cublasWorkspace;

        int32_t constexpr threads = 256;
        int64_t const queryElements = static_cast<int64_t>(batch) * length * channels;
        uint32_t const queryBlocks = static_cast<uint32_t>(
            std::min<int64_t>(65535, (queryElements + threads - 1) / threads));
        if (inputType == DataType::kHALF)
        {
            prepareQueriesAndPositionPointers<<<queryBlocks, threads, 0, stream>>>(
                static_cast<half const*>(inputs[0]), static_cast<half const*>(inputs[1]),
                static_cast<half const*>(inputs[2]), static_cast<half const*>(inputs[3]),
                static_cast<half*>(contentQuery), static_cast<half*>(positionQuery),
                static_cast<half*>(positionScores), positionPointers, queryPointers, outputPointers,
                batch, heads, length, relativeLength, channels, headDim);
        }
        else if (inputType == DataType::kBF16)
        {
            prepareQueriesAndPositionPointers<<<queryBlocks, threads, 0, stream>>>(
                static_cast<__nv_bfloat16 const*>(inputs[0]),
                static_cast<__nv_bfloat16 const*>(inputs[1]),
                static_cast<__nv_bfloat16 const*>(inputs[2]),
                static_cast<__nv_bfloat16 const*>(inputs[3]),
                static_cast<__nv_bfloat16*>(contentQuery),
                static_cast<__nv_bfloat16*>(positionQuery),
                static_cast<__nv_bfloat16*>(positionScores), positionPointers, queryPointers,
                outputPointers, batch, heads, length, relativeLength, channels, headDim);
        }
        else
        {
            prepareQueriesAndPositionPointers<<<queryBlocks, threads, 0, stream>>>(
                static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
                static_cast<float const*>(inputs[2]), static_cast<float const*>(inputs[3]),
                static_cast<float*>(contentQuery), static_cast<float*>(positionQuery),
                static_cast<float*>(positionScores), positionPointers, queryPointers,
                outputPointers, batch, heads, length, relativeLength, channels, headDim);
        }
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        float const alpha = 1.0F;
        float const beta = 0.0F;
        cublasComputeType_t const computeType = getCublasComputeType(mTactic);
        // cuBLAS is column-major. A contiguous row-major [T, D] query aliases a
        // column-major [D, T] matrix, so this product writes row-major [T, 2T-1]
        // position scores without materializing either transpose.
        cublasStatus_t status = cublasGemmBatchedEx(mCublas, CUBLAS_OP_T, CUBLAS_OP_N,
            relativeLength, length, headDim, &alpha, positionPointers, dataType, channels,
            queryPointers, dataType, headDim, &beta, outputPointers, dataType, relativeLength,
            pointerCount, computeType, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }

        uint32_t const pointerBlocks = static_cast<uint32_t>(
            std::min<int64_t>(65535, (static_cast<int64_t>(pointerCount) + threads - 1) / threads));
        initializeContentPointers<<<pointerBlocks, threads, 0, stream>>>(inputs[0], contentQuery,
            attentionWeights, positionPointers, queryPointers, outputPointers, batch, heads, length,
            channels, headDim, elementBytes);
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }
        // Q is packed head-major by the preparation kernel. K remains a view
        // into the fused NTC QKV projection, whose leading dimension is 3C.
        // Pointer-batched GEMM therefore avoids materializing K or transposing
        // either projection tensor.
        status =
            cublasGemmBatchedEx(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, length, length, headDim, &alpha,
                positionPointers, dataType, 3 * channels, queryPointers, dataType, headDim, &beta,
                outputPointers, dataType, length, pointerCount, computeType, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }

        if (inputType == DataType::kHALF)
        {
            if (!launchParakeetSoftmax<half>(positionScores, static_cast<int32_t const*>(inputs[4]),
                    attentionWeights, attentionWeights, batch, heads, length, relativeLength,
                    mParameters.scale, stream))
            {
                return 1;
            }
        }
        else if (inputType == DataType::kBF16)
        {
            if (!launchParakeetSoftmax<__nv_bfloat16>(positionScores,
                    static_cast<int32_t const*>(inputs[4]), attentionWeights, attentionWeights,
                    batch, heads, length, relativeLength, mParameters.scale, stream))
            {
                return 1;
            }
        }
        else
        {
            if (!launchParakeetSoftmax<float>(positionScores,
                    static_cast<int32_t const*>(inputs[4]), attentionWeights, attentionWeights,
                    batch, heads, length, relativeLength, mParameters.scale, stream))
            {
                return 1;
            }
        }

        initializeValuePointers<<<pointerBlocks, threads, 0, stream>>>(inputs[0], attentionWeights,
            outputs[0], positionPointers, queryPointers, outputPointers, batch, heads, length,
            channels, headDim, elementBytes);
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }
        // V remains in the fused NTC QKV projection. The 3C leading dimension
        // advances between frames, while C preserves contiguous NTC output.
        status =
            cublasGemmBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, headDim, length, length, &alpha,
                positionPointers, dataType, 3 * channels, queryPointers, dataType, length, &beta,
                outputPointers, dataType, channels, pointerCount, computeType, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        // Each execution context owns an independent cuBLAS handle and cached
        // stream/workspace state, so concurrent contexts cannot race.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFields; }

  private:
    friend class ParakeetFlashAttentionPluginCreator;

    void initializeFields() noexcept
    {
        mSerializedFields[0] = {kScaleField, &mParameters.scale, PluginFieldType::kFLOAT32, 1};
        mFields = {static_cast<int32_t>(mSerializedFields.size()), mSerializedFields.data()};
    }

    ParakeetFlashAttentionParameters mParameters{};
    cublasHandle_t mCublas{nullptr};
    cudaStream_t mStream{nullptr};
    void* mWorkspace{nullptr};
    int32_t mTactic{kStrictComputeTactic};
    DataType mInputType{DataType::kFLOAT};
    bool mInitialized{false};
    std::array<char, kTimingCacheIdSize> mTimingCacheId{};
    std::array<PluginField, 1> mSerializedFields{};
    PluginFieldCollection mFields{};
};

class ParakeetFlashAttentionPluginCreator final : public IPluginCreatorV3One
{
  public:
    ParakeetFlashAttentionPluginCreator() noexcept
    {
        mAttributes[0] = {kScaleField, nullptr, PluginFieldType::kFLOAT32, 1};
        mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
    }

    char const* getPluginName() const noexcept override { return kPluginName; }

    char const* getPluginVersion() const noexcept override { return kPluginVersion; }

    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fields,
        TensorRTPhase phase) noexcept override
    {
        static_cast<void>(name);
        static_cast<void>(phase);
        if (fields == nullptr || fields->fields == nullptr || fields->nbFields != 1)
        {
            return nullptr;
        }

        ParakeetFlashAttentionParameters parameters{};
        auto const& field = fields->fields[0];
        if (field.name == nullptr || std::string_view(field.name) != kScaleField
            || field.type != PluginFieldType::kFLOAT32 || field.length != 1
            || field.data == nullptr)
        {
            return nullptr;
        }
        parameters.scale = *static_cast<float const*>(field.data);
        if (!std::isfinite(parameters.scale) || parameters.scale <= 0.0F
            || !std::isfinite(parameters.scale * kLog2E))
        {
            return nullptr;
        }
        auto* plugin = new (std::nothrow) ParakeetFlashAttentionPlugin(parameters);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

  private:
    std::array<PluginField, 1> mAttributes{};
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrParakeetFlashAttentionPlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Builder and runtime use distinct registries. Treat an existing matching
    // creator as success so repeated package imports remain idempotent.
    static ParakeetFlashAttentionPluginCreator runtimeCreator;
    static ParakeetFlashAttentionPluginCreator builderCreator;
    auto ensureRegistered =
        [](IPluginRegistry* registry, ParakeetFlashAttentionPluginCreator& creator) noexcept
    {
        if (registry == nullptr)
        {
            return false;
        }
        if (registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace) != nullptr)
        {
            return true;
        }
        return registry->registerCreator(creator, kPluginNamespace)
               || registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace) != nullptr;
    };
    bool const runtimeRegistered = ensureRegistered(getPluginRegistry(), runtimeCreator);
    auto* builderRegistry =
        nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    bool const builderRegistered = ensureRegistered(builderRegistry, builderCreator);
    return runtimeRegistered && builderRegistered;
}
