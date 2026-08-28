// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "cublas_tactics.h"
#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

namespace
{

// Inputs use packed, contiguous layouts:
//   projection [N, T, H * (2 * query_dim + 4)]
//   position   [1, 2 * T - 1, H, 4]
//   mask       [N, T]
// The plugin produces normalized attention weights [N, H, T, T] while
// applying the Transformer-XL relative shift directly, without materializing
// query, key, position, or shifted-score transposes.
constexpr char const* kPluginName = "zipformer_relative_attention";
constexpr char const* kPluginVersion = "1";
constexpr int32_t kInputCount = 3;
constexpr int32_t kOutputCount = 1;
constexpr char const* kTimingCacheId =
    "layout=ntc;position_head_dim=4;padded_query_halo=7;softmax=v1";
constexpr size_t kCublasWorkspaceBytes = 16U << 20;
constexpr int32_t kPositionHeadDim = 4;
constexpr int32_t kPaddedQueryHalo = 7;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 4;
constexpr int32_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

constexpr int32_t warpRowBlocks(int32_t rows) noexcept
{
    return static_cast<int32_t>(
        (static_cast<int64_t>(rows) + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

constexpr bool isSupportedDataType(DataType type) noexcept
{
    return type == DataType::kFLOAT || type == DataType::kHALF
        || type == DataType::kBF16;
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

bool hasAddressableBytes(Dims const& dims, int32_t elementBytes) noexcept
{
    if (dims.nbDims < 1 || elementBytes < 1)
    {
        return false;
    }
    int64_t elements = 1;
    int64_t const maxElements =
        std::numeric_limits<int64_t>::max() / elementBytes;
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (dims.d[index] < 1
            || dims.d[index] > std::numeric_limits<int32_t>::max()
            || elements > maxElements / dims.d[index])
        {
            return false;
        }
        elements *= dims.d[index];
    }
    return true;
}

bool checkedMultiply(int64_t left, int64_t right, int64_t& output) noexcept
{
    if (left < 1 || right < 1
        || left > std::numeric_limits<int64_t>::max() / right)
    {
        return false;
    }
    output = left * right;
    return true;
}

bool haveValidShapes(Dims const& projection, Dims const& position,
    Dims const& mask, Dims const& scores, DataType type) noexcept
{
    // projection: [N, T, H * (2 * query_dim + 4)]
    // position:   [1, 2 * T - 1, H, 4]
    // mask:       [N, T]
    // scores:     [N, H, T, T]
    if (projection.nbDims != 3 || position.nbDims != 4 || mask.nbDims != 2
        || scores.nbDims != 4
        || !hasAddressableBytes(projection, dataTypeBytes(type))
        || !hasAddressableBytes(position, dataTypeBytes(type))
        || !hasAddressableBytes(mask, sizeof(bool))
        || !hasAddressableBytes(scores, dataTypeBytes(type))
        || position.d[0] != 1 || position.d[3] != kPositionHeadDim
        || mask.d[0] != projection.d[0] || mask.d[1] != projection.d[1]
        || scores.d[0] != projection.d[0] || scores.d[1] != position.d[2]
        || scores.d[2] != projection.d[1] || scores.d[3] != projection.d[1])
    {
        return false;
    }

    int64_t const batchSize = projection.d[0];
    int64_t const sequenceLength = projection.d[1];
    int64_t const projectionDim = projection.d[2];
    int64_t const numHeads = position.d[2];
    int64_t const expectedPositions = 2LL * sequenceLength - 1;
    int64_t batchHeads{};
    int64_t rows{};
    if (position.d[1] != expectedPositions || projectionDim % numHeads != 0
        || !checkedMultiply(batchSize, numHeads, batchHeads)
        || !checkedMultiply(batchHeads, sequenceLength, rows)
        || rows > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    int64_t const dimensionsPerHead = projectionDim / numHeads;
    return dimensionsPerHead > kPositionHeadDim
        && (dimensionsPerHead - kPositionHeadDim) % 2 == 0;
}

__device__ __forceinline__ float warpMaximum(float value)
{
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warpSum(float value)
{
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float toFloat(float value)
{
    return value;
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ half fromFloat(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat(float value)
{
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float fromFloat(float value)
{
    return value;
}

template <typename T>
__device__ float2 loadPair(T const* values);

template <>
__device__ __forceinline__ float2 loadPair(half const* values)
{
    return __half22float2(*reinterpret_cast<half2 const*>(values));
}

template <>
__device__ __forceinline__ float2 loadPair(__nv_bfloat16 const* values)
{
    return __bfloat1622float2(
        *reinterpret_cast<__nv_bfloat162 const*>(values));
}

template <>
__device__ __forceinline__ float2 loadPair(float const* values)
{
    return *reinterpret_cast<float2 const*>(values);
}

__device__ __forceinline__ float blockMaximum(float value, float* warpValues)
{
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const warp = threadIdx.x / kWarpSize;
    value = warpMaximum(value);
    if (lane == 0)
    {
        warpValues[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + kWarpSize - 1) / kWarpSize
        ? warpValues[lane]
        : -FLT_MAX;
    if (warp == 0)
    {
        value = warpMaximum(value);
    }
    if (threadIdx.x == 0)
    {
        warpValues[0] = value;
    }
    __syncthreads();
    return warpValues[0];
}

__device__ __forceinline__ float blockSum(float value, float* warpValues)
{
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const warp = threadIdx.x / kWarpSize;
    value = warpSum(value);
    if (lane == 0)
    {
        warpValues[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + kWarpSize - 1) / kWarpSize
        ? warpValues[lane]
        : 0.0F;
    if (warp == 0)
    {
        value = warpSum(value);
    }
    if (threadIdx.x == 0)
    {
        warpValues[0] = value;
    }
    __syncthreads();
    return warpValues[0];
}

template <typename T, int ValuesPerThread>
__global__ void softmaxWarpRows(T const* projection, T const* position,
    bool const* mask, T* scores, int32_t batchSize, int32_t numHeads,
    int32_t sequenceLength, int32_t queryDim)
{
    // Four warps independently normalize four (batch, head, query) rows. Each
    // lane owns strided key positions and keeps their logits in registers.
    int32_t const warp = threadIdx.x / kWarpSize;
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const row = blockIdx.x * kWarpsPerBlock + warp;
    int32_t const numRows = batchSize * numHeads * sequenceLength;
    if (row >= numRows)
    {
        return;
    }
    int32_t const query = row % sequenceLength;
    int32_t const head = (row / sequenceLength) % numHeads;
    int32_t const batch = row / (numHeads * sequenceLength);
    int32_t const projectionDim =
        numHeads * (2 * queryDim + kPositionHeadDim);
    int32_t const positionQueryOffset =
        2 * numHeads * queryDim + head * kPositionHeadDim;
    int64_t const projectionBase =
        (static_cast<int64_t>(batch) * sequenceLength + query) * projectionDim;
    // Zipformer masks a contiguous padded suffix. Preserve its established
    // seven-query halo, then skip attention work for deeper unused padded rows.
    if (query >= kPaddedQueryHalo
        && mask[batch * sequenceLength + query - kPaddedQueryHalo])
    {
        for (int32_t key = lane; key < sequenceLength; key += kWarpSize)
        {
            scores[static_cast<int64_t>(row) * sequenceLength + key]
                = fromFloat<T>(0.0F);
        }
        return;
    }

    // The projection row width and position-query offset are both even in
    // elements. CUDA allocations are at least 256-byte aligned, so every pair
    // remains naturally aligned for half2, bfloat162, and float2 loads.
    float2 const query01 = loadPair(projection + projectionBase + positionQueryOffset);
    float2 const query23 = loadPair(projection + projectionBase + positionQueryOffset + 2);
    int32_t const slots =
        (sequenceLength + kWarpSize - 1) / kWarpSize;
    float values[ValuesPerThread];
    float maximum = -FLT_MAX;
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        float value = -FLT_MAX;
        bool const valid = slot < slots && key < sequenceLength
            && !mask[batch * sequenceLength + key];
        if (valid)
        {
            // Apply the Transformer-XL relative shift directly for this
            // (query, key) pair instead of materializing a shifted tensor.
            int32_t const relative = sequenceLength - 1 - query + key;
            int64_t const positionBase =
                (static_cast<int64_t>(relative) * numHeads + head)
                * kPositionHeadDim;
            float2 const position01 = loadPair(position + positionBase);
            float2 const position23 = loadPair(position + positionBase + 2);
            value = toFloat(scores[static_cast<int64_t>(row) * sequenceLength + key])
                + query01.x * position01.x + query01.y * position01.y
                + query23.x * position23.x + query23.y * position23.y;
        }
        values[slot] = value;
        if (valid) maximum = fmaxf(maximum, value);
    }
    maximum = warpMaximum(maximum);
    maximum = __shfl_sync(0xFFFFFFFFU, maximum, 0);
    float sum = 0.0F;
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        bool const valid = slot < slots && key < sequenceLength
            && !mask[batch * sequenceLength + key];
        float const value = valid ? __expf(values[slot] - maximum) : 0.0F;
        values[slot] = value;
        sum += value;
    }
    sum = warpSum(sum);
    sum = __shfl_sync(0xFFFFFFFFU, sum, 0);
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = lane + slot * kWarpSize;
        if (slot < slots && key < sequenceLength)
        {
            scores[static_cast<int64_t>(row) * sequenceLength + key]
                = fromFloat<T>(sum > 0.0F ? values[slot] / sum : 0.0F);
        }
    }
}

template <typename T>
__global__ void softmaxBlockGeneric(T const* projection, T const* position,
    bool const* mask, T* scores, int32_t numHeads, int32_t sequenceLength,
    int32_t queryDim)
{
    // One block owns one row for sequences too long to retain every logit in
    // registers. The score buffer stages reduced-precision logits and exponentials
    // between the block-wide maximum and sum reductions.
    int32_t const row = blockIdx.x;
    int32_t const thread = threadIdx.x;
    int32_t const query = row % sequenceLength;
    int32_t const head = (row / sequenceLength) % numHeads;
    int32_t const batch = row / (numHeads * sequenceLength);
    int32_t const projectionDim =
        numHeads * (2 * queryDim + kPositionHeadDim);
    int32_t const positionQueryOffset =
        2 * numHeads * queryDim + head * kPositionHeadDim;
    int64_t const projectionBase =
        (static_cast<int64_t>(batch) * sequenceLength + query) * projectionDim;
    if (query >= kPaddedQueryHalo
        && mask[batch * sequenceLength + query - kPaddedQueryHalo])
    {
        for (int32_t key = thread; key < sequenceLength; key += blockDim.x)
        {
            scores[static_cast<int64_t>(row) * sequenceLength + key]
                = fromFloat<T>(0.0F);
        }
        return;
    }
    float2 const query01 = loadPair(projection + projectionBase + positionQueryOffset);
    float2 const query23 = loadPair(projection + projectionBase + positionQueryOffset + 2);
    int64_t const scoreBase = static_cast<int64_t>(row) * sequenceLength;
    float maximum = -FLT_MAX;
    for (int32_t key = thread; key < sequenceLength; key += blockDim.x)
    {
        int32_t const relative = sequenceLength - 1 - query + key;
        int64_t const positionBase =
            (static_cast<int64_t>(relative) * numHeads + head)
            * kPositionHeadDim;
        float2 const position01 = loadPair(position + positionBase);
        float2 const position23 = loadPair(position + positionBase + 2);
        bool const valid = !mask[batch * sequenceLength + key];
        float const value = valid
            ? toFloat(scores[scoreBase + key])
                + query01.x * position01.x + query01.y * position01.y
                + query23.x * position23.x + query23.y * position23.y
            : -FLT_MAX;
        scores[scoreBase + key] = fromFloat<T>(value);
        if (valid) maximum = fmaxf(maximum, value);
    }
    // Keep the two reductions in separate shared arrays. Reusing one array
    // would let a fast warp begin the sum reduction while another warp still
    // reads the maximum returned by blockMaximum().
    __shared__ float maximumReduction[kWarpSize];
    __shared__ float sumReduction[kWarpSize];
    maximum = blockMaximum(maximum, maximumReduction);
    float sum = 0.0F;
    for (int32_t key = thread; key < sequenceLength; key += blockDim.x)
    {
        float const value = mask[batch * sequenceLength + key]
            ? 0.0F : __expf(toFloat(scores[scoreBase + key]) - maximum);
        scores[scoreBase + key] = fromFloat<T>(value);
        sum += value;
    }
    sum = blockSum(sum, sumReduction);
    for (int32_t key = thread; key < sequenceLength; key += blockDim.x)
    {
        scores[scoreBase + key] = fromFloat<T>(
            sum > 0.0F ? toFloat(scores[scoreBase + key]) / sum : 0.0F);
    }
}

template <typename T, int ValuesPerThread>
__global__ void softmaxBlockRegisters(T const* projection, T const* position,
    bool const* mask, T* scores, int32_t numHeads, int32_t sequenceLength,
    int32_t queryDim)
{
    // One 512-thread block owns one row and keeps up to four logits per thread
    // in registers, avoiding two full score-buffer round trips for medium lengths.
    int32_t const row = blockIdx.x;
    int32_t const thread = threadIdx.x;
    int32_t const query = row % sequenceLength;
    int32_t const head = (row / sequenceLength) % numHeads;
    int32_t const batch = row / (numHeads * sequenceLength);
    int32_t const projectionDim =
        numHeads * (2 * queryDim + kPositionHeadDim);
    int32_t const positionQueryOffset =
        2 * numHeads * queryDim + head * kPositionHeadDim;
    int64_t const projectionBase =
        (static_cast<int64_t>(batch) * sequenceLength + query) * projectionDim;
    int64_t const scoreBase = static_cast<int64_t>(row) * sequenceLength;
    if (query >= kPaddedQueryHalo
        && mask[batch * sequenceLength + query - kPaddedQueryHalo])
    {
        for (int32_t key = thread; key < sequenceLength; key += blockDim.x)
        {
            scores[scoreBase + key] = fromFloat<T>(0.0F);
        }
        return;
    }

    float2 const query01 = loadPair(projection + projectionBase + positionQueryOffset);
    float2 const query23 = loadPair(projection + projectionBase + positionQueryOffset + 2);
    float values[ValuesPerThread];
    float maximum = -FLT_MAX;
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = thread + slot * blockDim.x;
        float value = -FLT_MAX;
        bool const valid = key < sequenceLength
            && !mask[batch * sequenceLength + key];
        if (valid)
        {
            int32_t const relative = sequenceLength - 1 - query + key;
            int64_t const positionBase =
                (static_cast<int64_t>(relative) * numHeads + head)
                * kPositionHeadDim;
            float2 const position01 = loadPair(position + positionBase);
            float2 const position23 = loadPair(position + positionBase + 2);
            value = toFloat(scores[scoreBase + key])
                + query01.x * position01.x + query01.y * position01.y
                + query23.x * position23.x + query23.y * position23.y;
        }
        // Match the generic path's reduced-precision materialization before softmax.
        value = toFloat(fromFloat<T>(value));
        values[slot] = value;
        if (valid) maximum = fmaxf(maximum, value);
    }

    // Maximum and sum reductions must not reuse shared storage between helper
    // calls because warps leave the helpers at different times.
    __shared__ float maximumReduction[kWarpSize];
    __shared__ float sumReduction[kWarpSize];
    maximum = blockMaximum(maximum, maximumReduction);
    float sum = 0.0F;
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = thread + slot * blockDim.x;
        if (key < sequenceLength)
        {
            float const value = mask[batch * sequenceLength + key]
                ? 0.0F : __expf(values[slot] - maximum);
            sum += value;
            values[slot] = toFloat(fromFloat<T>(value));
        }
    }
    sum = blockSum(sum, sumReduction);
#pragma unroll
    for (int slot = 0; slot < ValuesPerThread; ++slot)
    {
        int32_t const key = thread + slot * blockDim.x;
        if (key < sequenceLength)
        {
            scores[scoreBase + key]
                = fromFloat<T>(sum > 0.0F ? values[slot] / sum : 0.0F);
        }
    }
}

template <typename T>
constexpr cudaDataType_t cudaDataType();

template <>
constexpr cudaDataType_t cudaDataType<half>()
{
    return CUDA_R_16F;
}

template <>
constexpr cudaDataType_t cudaDataType<__nv_bfloat16>()
{
    return CUDA_R_16BF;
}

template <>
constexpr cudaDataType_t cudaDataType<float>()
{
    return CUDA_R_32F;
}

template <typename T>
int32_t enqueueRelativeAttention(cublasHandle_t cublas, int32_t tactic,
    PluginTensorDesc const* inputDesc, void const* const* inputs,
    void* const* outputs, cudaStream_t stream) noexcept
{
    int32_t const batchSize = inputDesc[0].dims.d[0];
    int32_t const sequenceLength = inputDesc[0].dims.d[1];
    int32_t const projectionDim = inputDesc[0].dims.d[2];
    int32_t const numHeads = inputDesc[1].dims.d[2];
    int32_t const queryDim =
        (projectionDim / numHeads - kPositionHeadDim) / 2;

    float const alpha = 1.0F;
    float const beta = 0.0F;
    int64_t const projectionBatchStride
        = static_cast<int64_t>(sequenceLength) * projectionDim;
    int64_t const outputBatchStride
        = static_cast<int64_t>(numHeads) * sequenceLength * sequenceLength;
    auto const* projection = static_cast<T const*>(inputs[0]);
    auto* output = static_cast<T*>(outputs[0]);
    int32_t const rows = batchSize * numHeads * sequenceLength;
    auto const* position = static_cast<T const*>(inputs[1]);
    auto const* mask = static_cast<bool const*>(inputs[2]);
    // Query and key heads are interleaved in the projection's final dimension.
    // cuBLAS interprets each strided NTC head slice as a column-major [D, T]
    // matrix. K^T Q therefore lands directly in the row-major [query, key]
    // output storage without materializing either transpose. Each GEMM batches
    // one head over all utterances.
    for (int32_t head = 0; head < numHeads; ++head)
    {
        auto const* query = projection + head * queryDim;
        auto const* key =
            projection + numHeads * queryDim + head * queryDim;
        auto* headOutput = output
            + static_cast<int64_t>(head) * sequenceLength * sequenceLength;
        if (cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                sequenceLength, sequenceLength, queryDim, &alpha, key,
                cudaDataType<T>(), projectionDim, projectionBatchStride,
                query, cudaDataType<T>(), projectionDim,
                projectionBatchStride, &beta, headOutput, cudaDataType<T>(),
                sequenceLength, outputBatchStride, batchSize,
                getCublasComputeType(tactic), CUBLAS_GEMM_DEFAULT)
            != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }
    }

    // Short rows stay within one warp. Medium rows use one block and retain
    // logits in registers. Only long rows stage intermediates through the
    // output buffer for block-wide reductions.
    if (sequenceLength <= 384)
    {
        softmaxWarpRows<T, 12><<<
            warpRowBlocks(rows),
            kThreadsPerBlock, 0, stream>>>(projection, position, mask, output,
            batchSize, numHeads, sequenceLength, queryDim);
    }
    else if (sequenceLength <= 512)
    {
        softmaxWarpRows<T, 16><<<
            warpRowBlocks(rows),
            kThreadsPerBlock, 0, stream>>>(projection, position, mask, output,
            batchSize, numHeads, sequenceLength, queryDim);
    }
    else if (sequenceLength <= 1024)
    {
        softmaxWarpRows<T, 32><<<
            warpRowBlocks(rows),
            kThreadsPerBlock, 0, stream>>>(projection, position, mask, output,
            batchSize, numHeads, sequenceLength, queryDim);
    }
    else if (sequenceLength <= 2048)
    {
        softmaxBlockRegisters<T, 4><<<rows, 512, 0, stream>>>(
            projection, position, mask, output, numHeads, sequenceLength,
            queryDim);
    }
    else
    {
        int32_t threads = kWarpSize;
        while (threads * 3 < sequenceLength && threads < 512)
        {
            threads *= 2;
        }
        softmaxBlockGeneric<<<rows, threads, 0, stream>>>(
            projection, position, mask, output, numHeads, sequenceLength,
            queryDim);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

// Compute normalized Transformer-XL relative-attention scores directly from the
// packed NTC projection and compact positional projection. Content scores use
// cuBLAS; custom CUDA kernels add relative scores, apply the key mask, and
// normalize rows without materializing query/key/position transposes. The mask
// must describe the contiguous padded suffix produced by the Zipformer encoder.
class RelativeAttentionPlugin final : public IPluginV3,
                                      public IPluginV3OneCore,
                                      public IPluginV3OneBuild,
                                      public IPluginV3OneRuntime
{
public:
    explicit RelativeAttentionPlugin(
        int32_t tactic = kStrictComputeTactic) noexcept
        : mTactic(tactic)
    {
        mInitialized = cublasCreate(&mCublas) == CUBLAS_STATUS_SUCCESS;
    }

    ~RelativeAttentionPlugin() override
    {
        if (mCublas != nullptr)
        {
            cublasDestroy(mCublas);
        }
    }

    IPluginCapability* getCapabilityInterface(
        PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime*>(this);
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore*>(this);
        default: return nullptr;
        }
    }

    IPluginV3* clone() noexcept override
    {
        auto* plugin = new (std::nothrow) RelativeAttentionPlugin(mTactic);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        plugin->mInputType = mInputType;
        return plugin;
    }

    char const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kPluginNamespace;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return kOutputCount;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        // TensorRT may propagate conservative, independently bounded min/max
        // dimensions through the surrounding symbolic graph. Validate only
        // profile-invariant metadata here; onShapeChange validates exact shapes.
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !mInitialized
            || inputs[0].desc.dims.nbDims != 3
            || inputs[1].desc.dims.nbDims != 4
            || inputs[2].desc.dims.nbDims != 2
            || outputs[0].desc.dims.nbDims != 4
            || inputs[0].desc.type != inputs[1].desc.type
            || outputs[0].desc.type != inputs[0].desc.type
            || !isSupportedDataType(inputs[0].desc.type)
            || inputs[2].desc.type != DataType::kBOOL
            || inputs[0].desc.format != TensorFormat::kLINEAR
            || inputs[1].desc.format != TensorFormat::kLINEAR
            || inputs[2].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR
            || !hasAddressableBytes(
                inputs[0].max, dataTypeBytes(inputs[0].desc.type))
            || !hasAddressableBytes(
                inputs[1].max, dataTypeBytes(inputs[1].desc.type))
            || !hasAddressableBytes(inputs[2].max, sizeof(bool))
            || !hasAddressableBytes(
                outputs[0].max, dataTypeBytes(outputs[0].desc.type)))
        {
            return 1;
        }
        mInputType = inputs[0].desc.type;
        return 0;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbInputs != kInputCount || nbOutputs != kOutputCount
            || !isSupportedDataType(inputTypes[0])
            || inputTypes[1] != inputTypes[0]
            || inputTypes[2] != DataType::kBOOL)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        static_cast<void>(shapeInputs);
        static_cast<void>(exprBuilder);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || nbShapeInputs != 0
            || inputs[0].nbDims != 3
            || inputs[1].nbDims != 4 || inputs[2].nbDims != 2
            || inputs[0].d[0] == nullptr || inputs[0].d[1] == nullptr
            || inputs[1].d[2] == nullptr)
        {
            return 1;
        }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[1].d[2];
        outputs[0].d[2] = inputs[0].d[1];
        outputs[0].d[3] = inputs[0].d[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos,
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || pos < 0
            || pos >= kInputCount + kOutputCount)
        {
            return false;
        }
        auto const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 2)
        {
            return desc.type == DataType::kBOOL;
        }
        return isSupportedDataType(desc.type)
            && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount)
        {
            return 0;
        }
        // TensorRT owns this scratch allocation. Binding it to the cuBLAS
        // handle avoids internal allocations and keeps execution capturable.
        return kCublasWorkspaceBytes;
    }

    int32_t getNbTactics() noexcept override
    {
        return getCublasComputeTacticCount(mInputType);
    }

    int32_t getValidTactics(int32_t* tactics,
        int32_t nbTactics) noexcept override
    {
        return writeCublasComputeTactics(tactics, nbTactics, mInputType);
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        return setCublasComputeTactic(tactic, mTactic, mInputType);
    }

    char const* getTimingCacheID() noexcept override
    {
        // TensorRT combines this implementation identity with concrete tensor
        // shapes and formats, allowing equivalent encoder layers to reuse
        // their dtype-specific cuBLAS tactic timings.
        return kTimingCacheId;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !mInitialized
            || inputs[0].type != inputs[1].type
            || outputs[0].type != inputs[0].type
            || !isSupportedDataType(inputs[0].type)
            || inputs[2].type != DataType::kBOOL
            || inputs[0].format != TensorFormat::kLINEAR
            || inputs[1].format != TensorFormat::kLINEAR
            || inputs[2].format != TensorFormat::kLINEAR
            || outputs[0].format != TensorFormat::kLINEAR
            || !isCublasComputeTactic(mTactic, inputs[0].type)
            || !haveValidShapes(inputs[0].dims, inputs[1].dims,
                inputs[2].dims, outputs[0].dims, inputs[0].type))
        {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr
            || inputs[2] == nullptr || outputs[0] == nullptr
            || workspace == nullptr
            || onShapeChange(inputDesc, kInputCount, outputDesc, kOutputCount) != 0)
        {
            return 1;
        }

        // Preserve errors from earlier asynchronous work instead of silently
        // attributing success to this invocation.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        bool const streamChanged = stream != mStream;
        if (streamChanged)
        {
            if (cublasSetStream(mCublas, stream) != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mStream = stream;
            // cublasSetStream resets the handle workspace, even when TensorRT
            // subsequently supplies the same address used by the previous stream.
            mWorkspace = nullptr;
        }
        if (workspace != mWorkspace)
        {
            if (cublasSetWorkspace(mCublas, workspace, kCublasWorkspaceBytes)
                != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mWorkspace = workspace;
        }

        switch (inputDesc[0].type)
        {
        case DataType::kHALF:
            return enqueueRelativeAttention<half>(mCublas, mTactic, inputDesc,
                inputs, outputs, stream);
        case DataType::kBF16:
            return enqueueRelativeAttention<__nv_bfloat16>(mCublas, mTactic,
                inputDesc, inputs, outputs, stream);
        case DataType::kFLOAT:
            return enqueueRelativeAttention<float>(mCublas, mTactic, inputDesc,
                inputs, outputs, stream);
        default: return 1;
        }
    }

    IPluginV3* attachToContext(
        IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        // Every execution context receives an independent handle and cached
        // stream/workspace state, so concurrent contexts cannot race.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    friend class RelativeAttentionCreator;

    cublasHandle_t mCublas{nullptr};
    cudaStream_t mStream{nullptr};
    void* mWorkspace{nullptr};
    DataType mInputType{DataType::kFLOAT};
    int32_t mTactic{kStrictComputeTactic};
    bool mInitialized{false};
    PluginFieldCollection mFields{};
};

class RelativeAttentionCreator final : public IPluginCreatorV3One
{
public:
    char const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kPluginNamespace;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFields;
    }

    IPluginV3* createPlugin(char const* name,
        PluginFieldCollection const* fields,
        TensorRTPhase phase) noexcept override
    {
        static_cast<void>(name);
        static_cast<void>(phase);
        if (fields == nullptr || fields->nbFields != 0)
        {
            return nullptr;
        }
        auto* plugin = new (std::nothrow) RelativeAttentionPlugin();
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

private:
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrZipformerRelativeAttentionPlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Builder and runtime use distinct registries. Repeated package imports are
    // successful when the matching creator has already been registered.
    static RelativeAttentionCreator runtimeCreator;
    static RelativeAttentionCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry,
                                RelativeAttentionCreator& creator) noexcept {
        if (registry == nullptr)
        {
            return false;
        }
        if (registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace)
            != nullptr)
        {
            return true;
        }
        return registry->registerCreator(creator, kPluginNamespace)
            || registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace)
                != nullptr;
    };
    bool const runtimeRegistered =
        ensureRegistered(getPluginRegistry(), runtimeCreator);
    auto* builderRegistry =
        nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    bool const builderRegistered = ensureRegistered(builderRegistry, builderCreator);
    return runtimeRegistered && builderRegistered;
}
