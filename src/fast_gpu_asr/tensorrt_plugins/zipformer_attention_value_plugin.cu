// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string_view>

#include "cublas_tactics.h"
#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

namespace
{

// Compute O[n, q, h, d] = sum_k A[n, h, q, k] * V[n, k, h, d].
// Attention uses NHTT storage while values and output stay in NTC storage.
// TensorRT times a warp-specialized head-dimension-12 kernel against several
// cuBLAS tactics and serializes the fastest compatible choice in the engine.
constexpr char const* kPluginName = "zipformer_attention_value";
constexpr char const* kPluginVersion = "1";
constexpr char const* kNumHeadsField = "num_heads";
constexpr int32_t kInputCount = 2;
constexpr int32_t kOutputCount = 1;
constexpr int32_t kNarrowHeadTactic = 5;
constexpr int32_t kNarrowHeadDim = 12;
constexpr int32_t kPointerThreads = 256;
constexpr int32_t kMaxBlocks = 65535;
constexpr size_t kTimingCacheIdSize = 32;

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

bool haveValidShapes(Dims const& attention, Dims const& value,
    Dims const& output, int32_t valueHeads, int32_t elementBytes) noexcept
{
    if (valueHeads < 1 || attention.nbDims != 4 || value.nbDims != 3
        || output.nbDims != 3 || !hasAddressableBytes(attention, elementBytes)
        || !hasAddressableBytes(value, elementBytes)
        || !hasAddressableBytes(output, elementBytes)
        || attention.d[1] < valueHeads
        || (valueHeads > 1 && attention.d[1] != valueHeads)
        || attention.d[0] != value.d[0]
        || attention.d[2] != attention.d[3]
        || attention.d[2] != value.d[1] || value.d[2] % valueHeads != 0
        || attention.d[0] > std::numeric_limits<int32_t>::max() / valueHeads
        || output.d[0] != value.d[0] || output.d[1] != value.d[1]
        || output.d[2] != value.d[2])
    {
        return false;
    }
    return true;
}

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat(float value)
{
    return value;
}

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

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat(float value)
{
    return value;
}

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

template <typename T>
__global__ void attentionValueHead12(T const* attentionWeights,
    T const* value, T* output, int32_t batchSize, int32_t attentionHeads,
    int32_t valueHeads, int32_t sequenceLength, int32_t channels)
{
    // One warp owns one (batch, value head, query) row. Its lanes divide the
    // key dimension, accumulate all 12 channels in FP32, and reduce in-warp.
    int32_t const lane = threadIdx.x % warpSize;
    int64_t const warp =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x)
        / warpSize;
    int64_t const warpStride =
        static_cast<int64_t>(blockDim.x) * gridDim.x / warpSize;
    int64_t const rowCount = static_cast<int64_t>(batchSize) * valueHeads
        * sequenceLength;

    for (int64_t row = warp; row < rowCount; row += warpStride)
    {
        // Decode the flattened work item into the output row owned by this
        // warp. Grid-stride iteration covers profiles larger than CUDA's grid
        // limit without splitting a row across warps.
        int32_t const query = static_cast<int32_t>(row % sequenceLength);
        int32_t const headBatch = static_cast<int32_t>(row / sequenceLength);
        int32_t const head = headBatch % valueHeads;
        int32_t const batch = headBatch / valueHeads;
        int64_t const attentionOffset =
            (static_cast<int64_t>(batch) * attentionHeads + head)
            * sequenceLength * sequenceLength
            + static_cast<int64_t>(query) * sequenceLength;
        int64_t const valueOffset =
            static_cast<int64_t>(batch) * sequenceLength * channels
            + head * kNarrowHeadDim;

        float sums[kNarrowHeadDim]{};
        for (int64_t key = lane; key < sequenceLength; key += warpSize)
        {
            float const weight =
                toFloat(attentionWeights[attentionOffset + key]);
            int64_t const keyOffset = valueOffset + key * channels;
#pragma unroll
            for (int32_t channel = 0; channel < kNarrowHeadDim; ++channel)
            {
                sums[channel] += weight
                    * toFloat(value[keyOffset + channel]);
            }
        }

        // Fold the per-lane partial sums into lane zero entirely through
        // registers. All lanes remain active, including lanes with no key when
        // sequenceLength is shorter than a warp.
        for (int32_t offset = warpSize / 2; offset > 0; offset /= 2)
        {
#pragma unroll
            for (int32_t channel = 0; channel < kNarrowHeadDim; ++channel)
            {
                sums[channel] += __shfl_down_sync(
                    0xFFFFFFFFU, sums[channel], offset);
            }
        }

        if (lane == 0)
        {
            int64_t const outputOffset =
                (static_cast<int64_t>(batch) * sequenceLength + query)
                    * channels
                + head * kNarrowHeadDim;
#pragma unroll
            for (int32_t channel = 0; channel < kNarrowHeadDim; ++channel)
            {
                output[outputOffset + channel] =
                    fromFloat<T>(sums[channel]);
            }
        }
    }
}

__global__ void initializeAttentionValuePointers(void const* attentionWeights,
    void const* value, void* output, void const** attentionPointers,
    void const** valuePointers, void** outputPointers, int32_t batchCount,
    int32_t attentionHeads, int32_t valueHeads, int32_t sequenceLength,
    int32_t channels, int32_t headDim, int32_t elementBytes)
{
    // Each pointer triple describes one independent (batch, value-head) GEMM.
    // Head indices map one-to-one except for nonlinear attention, whose single
    // value head intentionally consumes attention head zero. Values remain
    // interleaved in NTC; cuBLAS receives `channels` as the leading dimension
    // and therefore steps between keys without a transpose.
    int64_t const thread = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    int64_t const threadStride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = thread; index < batchCount; index += threadStride)
    {
        int32_t const batch = static_cast<int32_t>(index / valueHeads);
        int32_t const head = static_cast<int32_t>(index - batch * valueHeads);
        auto const* attentionBytes =
            static_cast<unsigned char const*>(attentionWeights);
        auto const* valueBytes = static_cast<unsigned char const*>(value);
        auto* outputBytes = static_cast<unsigned char*>(output);
        int64_t const valueOffset =
            (static_cast<int64_t>(batch) * sequenceLength * channels + head * headDim)
                * elementBytes;
        attentionPointers[index] = attentionBytes
            + (static_cast<int64_t>(batch) * attentionHeads + head)
                * sequenceLength * sequenceLength * elementBytes;
        valuePointers[index] = valueBytes + valueOffset;
        outputPointers[index] = outputBytes + valueOffset;
    }
}

// Compute output[n, q, h, d] = sum_k attention[n, h, q, k] * value[n, k, h, d].
// The head-12 tactic handles Zipformer self-attention directly. Other shapes,
// including single-value-head nonlinear attention, use batched cuBLAS GEMMs
// with device pointer arrays so all tensors remain in NTC-compatible storage.
class AttentionValuePlugin final : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
{
public:
    explicit AttentionValuePlugin(
        int32_t valueHeads = 1, int32_t tactic = kStrictComputeTactic,
        bool supportsNarrowHeadTactic = false) noexcept
        : mValueHeads(valueHeads), mTactic(tactic),
          mSupportsNarrowHeadTactic(supportsNarrowHeadTactic)
    {
        // Equivalent plugin instances share TensorRT timing results. Include
        // valueHeads because it changes both the GEMM batch and head dimension.
        constexpr std::string_view prefix{"num_heads="};
        std::copy(prefix.begin(), prefix.end(), mTimingCacheId.begin());
        auto const result = std::to_chars(
            mTimingCacheId.data() + prefix.size(),
            mTimingCacheId.data() + mTimingCacheId.size() - 1, valueHeads);
        bool const validTimingCacheId = result.ec == std::errc{};
        if (validTimingCacheId)
        {
            *result.ptr = '\0';
        }
        mSerializedField = {
            kNumHeadsField, &mValueHeads, PluginFieldType::kINT32, 1};
        mFields = {1, &mSerializedField};
        mInitialized = valueHeads > 0 && validTimingCacheId
            && cublasCreate(&mHandle) == CUBLAS_STATUS_SUCCESS;
    }

    ~AttentionValuePlugin() override
    {
        if (mHandle != nullptr)
        {
            cublasDestroy(mHandle);
        }
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
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
        auto* plugin = new (std::nothrow) AttentionValuePlugin(
            mValueHeads, mTactic, mSupportsNarrowHeadTactic);
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

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || inputs[0].desc.dims.nbDims != 4
            || inputs[1].desc.dims.nbDims != 3
            || outputs[0].desc.dims.nbDims != 3 || mValueHeads < 1
            || inputs[0].desc.type != inputs[1].desc.type
            || outputs[0].desc.type != inputs[1].desc.type
            || !isSupportedDataType(inputs[1].desc.type)
            || inputs[0].desc.format != TensorFormat::kLINEAR
            || inputs[1].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR
            || !haveValidShapes(inputs[0].min, inputs[1].min, outputs[0].min,
                mValueHeads, dataTypeBytes(inputs[1].desc.type))
            || !haveValidShapes(inputs[0].opt, inputs[1].opt, outputs[0].opt,
                mValueHeads, dataTypeBytes(inputs[1].desc.type))
            || !haveValidShapes(inputs[0].max, inputs[1].max, outputs[0].max,
                mValueHeads, dataTypeBytes(inputs[1].desc.type))
            || !mInitialized)
        {
            return 1;
        }
        int64_t const channels = inputs[1].desc.dims.d[2];
        // The custom kernel is profitable only for multi-head Zipformer
        // self-attention, whose checkpoint fixes every value head at 12
        // channels. Nonlinear attention keeps the general cuBLAS path.
        mSupportsNarrowHeadTactic = mValueHeads > 1 && channels > 0
            && channels % mValueHeads == 0
            && channels / mValueHeads == kNarrowHeadDim;
        mInputType = inputs[1].desc.type;
        return 0;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes,
        int32_t nbOutputs,
        DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbInputs != kInputCount || nbOutputs != kOutputCount)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[1];
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs,
        int32_t nbInputs,
        DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        DimsExprs* outputs,
        int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        static_cast<void>(shapeInputs);
        static_cast<void>(exprBuilder);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || nbShapeInputs != 0)
        {
            return 1;
        }
        outputs[0] = inputs[1];
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos,
        DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || pos < 0
            || pos >= kInputCount + kOutputCount)
        {
            return false;
        }
        auto const& desc = inOut[pos].desc;
        return desc.format == TensorFormat::kLINEAR
            && isSupportedDataType(desc.type)
            && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || mValueHeads < 1
            || inputs[1].max.d[0] < 1
            || inputs[1].max.d[0]
                > std::numeric_limits<int32_t>::max() / mValueHeads)
        {
            return 0;
        }
        size_t const maxBatchSize = static_cast<size_t>(inputs[1].max.d[0]);
        size_t const valueHeads = static_cast<size_t>(mValueHeads);
        // The cuBLAS path stores three device pointer arrays consecutively:
        // attention inputs, value inputs, and outputs.
        constexpr size_t bytesPerPointerSet = sizeof(void*) * 3;
        if (maxBatchSize
            > std::numeric_limits<size_t>::max() / valueHeads
                / bytesPerPointerSet)
        {
            return 0;
        }
        return maxBatchSize * valueHeads * bytesPerPointerSet;
    }

    int32_t getNbTactics() noexcept override
    {
        return getCublasComputeTacticCount(mInputType, deviceSupportsAmpereCompute())
            + static_cast<int32_t>(mSupportsNarrowHeadTactic);
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        bool const ampereCompute = deviceSupportsAmpereCompute();
        int32_t const cublasTactics = getCublasComputeTacticCount(mInputType, ampereCompute);
        if (tactics == nullptr
            || nbTactics != cublasTactics
                + static_cast<int32_t>(mSupportsNarrowHeadTactic)
            || writeCublasComputeTactics(
                tactics, cublasTactics, mInputType, ampereCompute) != 0)
        {
            return 1;
        }
        if (mSupportsNarrowHeadTactic)
        {
            tactics[cublasTactics] = kNarrowHeadTactic;
        }
        return 0;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic == kNarrowHeadTactic)
        {
            // Runtime deserialization restores the tactic before the concrete
            // shape reaches onShapeChange(), where compatibility is checked.
            mTactic = tactic;
            return 0;
        }
        return setCublasComputeTactic(tactic, mTactic, mInputType);
    }

    char const* getTimingCacheID() noexcept override
    {
        return mTimingCacheId.data();
    }

    int32_t onShapeChange(
        PluginTensorDesc const* inputs,
        int32_t nbInputs,
        PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        // A single value head is used by nonlinear attention and intentionally
        // consumes the first of potentially several attention heads. For
        // ordinary self-attention, attention and value head counts must match.
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || mValueHeads < 1 || !mInitialized
            || inputs[0].type != inputs[1].type
            || outputs[0].type != inputs[1].type
            || !isSupportedDataType(inputs[1].type)
            || inputs[0].format != TensorFormat::kLINEAR
            || inputs[1].format != TensorFormat::kLINEAR
            || outputs[0].format != TensorFormat::kLINEAR
            || !haveValidShapes(inputs[0].dims, inputs[1].dims,
                outputs[0].dims, mValueHeads, dataTypeBytes(inputs[1].type))
            || (mTactic != kNarrowHeadTactic
                && !isCublasComputeTactic(mTactic, inputs[1].type))
            || (mTactic == kNarrowHeadTactic
                && (mValueHeads == 1
                    || inputs[1].dims.d[2] / mValueHeads != kNarrowHeadDim)))
        {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(
        PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr
            || outputs[0] == nullptr || !mInitialized
            || inputDesc[0].type != inputDesc[1].type
            || outputDesc[0].type != inputDesc[1].type
            || inputDesc[0].format != TensorFormat::kLINEAR
            || inputDesc[1].format != TensorFormat::kLINEAR
            || outputDesc[0].format != TensorFormat::kLINEAR
            || !haveValidShapes(inputDesc[0].dims, inputDesc[1].dims,
                outputDesc[0].dims, mValueHeads,
                dataTypeBytes(inputDesc[1].type))
            || (mTactic != kNarrowHeadTactic
                && !isCublasComputeTactic(mTactic, inputDesc[1].type))
            || (mTactic == kNarrowHeadTactic
                && (mValueHeads == 1
                    || inputDesc[1].dims.d[2] / mValueHeads
                        != kNarrowHeadDim)))
        {
            return 1;
        }

        auto const& attentionDims = inputDesc[0].dims;
        auto const& valueDims = inputDesc[1].dims;
        int32_t const batchSize = attentionDims.d[0];
        int32_t const attentionHeads = attentionDims.d[1];
        int32_t const sequenceLength = attentionDims.d[2];
        int32_t const channels = valueDims.d[2];
        int32_t const headDim = channels / mValueHeads;
        int64_t const batchCount64 =
            static_cast<int64_t>(batchSize) * mValueHeads;
        if (batchCount64 < 1
            || batchCount64 > std::numeric_limits<int32_t>::max())
        {
            return 1;
        }
        int32_t const batchCount = static_cast<int32_t>(batchCount64);

        // Preserve errors from earlier asynchronous work instead of silently
        // attributing success to this invocation.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        if (mTactic == kNarrowHeadTactic)
        {
            constexpr int32_t kThreads = 128;
            constexpr int32_t kWarpsPerBlock = kThreads / 32;
            // Four warps per block process four independent output rows. The
            // kernel's grid-stride loop handles any rows beyond 65,535 blocks.
            int64_t const rows = static_cast<int64_t>(batchCount) * sequenceLength;
            int32_t const blocks = static_cast<int32_t>(std::min<int64_t>(
                kMaxBlocks, rows / kWarpsPerBlock
                    + (rows % kWarpsPerBlock != 0)));
            if (inputDesc[1].type == DataType::kFLOAT)
            {
                attentionValueHead12<<<blocks, kThreads, 0, stream>>>(
                    static_cast<float const*>(inputs[0]),
                    static_cast<float const*>(inputs[1]),
                    static_cast<float*>(outputs[0]), batchSize, attentionHeads,
                    mValueHeads, sequenceLength, channels);
            }
            else if (inputDesc[1].type == DataType::kHALF)
            {
                attentionValueHead12<<<blocks, kThreads, 0, stream>>>(
                    static_cast<half const*>(inputs[0]),
                    static_cast<half const*>(inputs[1]),
                    static_cast<half*>(outputs[0]), batchSize, attentionHeads,
                    mValueHeads, sequenceLength, channels);
            }
            else if (inputDesc[1].type == DataType::kBF16)
            {
                attentionValueHead12<<<blocks, kThreads, 0, stream>>>(
                    static_cast<__nv_bfloat16 const*>(inputs[0]),
                    static_cast<__nv_bfloat16 const*>(inputs[1]),
                    static_cast<__nv_bfloat16*>(outputs[0]), batchSize,
                    attentionHeads, mValueHeads, sequenceLength, channels);
            }
            else
            {
                return 1;
            }
            return cudaGetLastError() == cudaSuccess ? 0 : 1;
        }

        if (stream != mStream)
        {
            // cublasSetStream mutates handle state. Avoid repeating it across
            // the 57 plugin invocations that normally share one encoder stream.
            if (cublasSetStream(mHandle, stream) != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mStream = stream;
        }

        float const alpha = 1.0F;
        float const beta = 0.0F;
        cudaDataType_t dataType{};
        int32_t const elementBytes = dataTypeBytes(inputDesc[1].type);
        switch (inputDesc[1].type)
        {
        case DataType::kFLOAT:
            dataType = CUDA_R_32F;
            break;
        case DataType::kHALF:
            dataType = CUDA_R_16F;
            break;
        case DataType::kBF16:
            dataType = CUDA_R_16BF;
            break;
        default: return 1;
        }
        if (workspace == nullptr)
        {
            return 1;
        }

        // NTC keeps each head dimension contiguous but gives flattened (batch,
        // head) matrices a nonuniform base stride at batch boundaries. Device
        // pointer arrays preserve NTC end to end without transpose kernels.
        auto* workspaceBytes = static_cast<unsigned char*>(workspace);
        auto** attentionPointers = reinterpret_cast<void const**>(workspaceBytes);
        auto** valuePointers = reinterpret_cast<void const**>(
            workspaceBytes + static_cast<size_t>(batchCount) * sizeof(void*));
        auto** outputPointers = reinterpret_cast<void**>(
            workspaceBytes + static_cast<size_t>(batchCount) * sizeof(void*) * 2);
        int32_t const pointerBlocks = static_cast<int32_t>(std::min<int64_t>(
            kMaxBlocks,
            (batchCount64 + kPointerThreads - 1) / kPointerThreads));
        initializeAttentionValuePointers<<<
            pointerBlocks, kPointerThreads, 0, stream>>>(
            inputs[0], inputs[1], outputs[0], attentionPointers, valuePointers,
            outputPointers, batchCount, attentionHeads, mValueHeads, sequenceLength,
            channels, headDim, elementBytes);
        if (cudaGetLastError() != cudaSuccess)
        {
            return 1;
        }

        // cuBLAS is column-major: NTC value storage is viewed as [headDim, T]
        // with leading dimension C, and row-major [T, T] attention is viewed
        // as its column-major transpose. The product is consequently written
        // directly into the corresponding NTC output head without copies.
        cublasStatus_t const status = cublasGemmBatchedEx(
            mHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            headDim,
            sequenceLength,
            sequenceLength,
            &alpha,
            valuePointers,
            dataType,
            channels,
            attentionPointers,
            dataType,
            sequenceLength,
            &beta,
            outputPointers,
            dataType,
            channels,
            batchCount,
            getCublasComputeType(mTactic),
            CUBLAS_GEMM_DEFAULT);
        return status == CUBLAS_STATUS_SUCCESS && cudaGetLastError() == cudaSuccess
            ? 0
            : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        // Give every TensorRT execution context an independent cuBLAS handle
        // and cached stream; sharing either would make concurrent contexts race.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    friend class AttentionValuePluginCreator;

    cublasHandle_t mHandle{nullptr};
    cudaStream_t mStream{nullptr};
    DataType mInputType{DataType::kFLOAT};
    int32_t mValueHeads{1};
    int32_t mTactic{kStrictComputeTactic};
    bool mInitialized{false};
    bool mSupportsNarrowHeadTactic{false};
    std::array<char, kTimingCacheIdSize> mTimingCacheId{};
    PluginField mSerializedField{};
    PluginFieldCollection mFields{};
};

class AttentionValuePluginCreator final : public IPluginCreatorV3One
{
public:
    AttentionValuePluginCreator()
    {
        mAttribute = {kNumHeadsField, nullptr, PluginFieldType::kINT32, 1};
        mFields = {1, &mAttribute};
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

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFields;
    }

    IPluginV3* createPlugin(
        char const* name,
        PluginFieldCollection const* fields,
        TensorRTPhase phase) noexcept override
    {
        static_cast<void>(name);
        static_cast<void>(phase);
        if (fields == nullptr || fields->nbFields < 0
            || (fields->nbFields > 0 && fields->fields == nullptr))
        {
            return nullptr;
        }
        int32_t valueHeads = 0;
        bool foundValueHeads = false;
        // Reject malformed and duplicate attributes instead of relying on the
        // ONNX parser to provide a well-formed field collection.
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name != nullptr
                && std::string_view(field.name) == kNumHeadsField)
            {
                if (foundValueHeads || field.type != PluginFieldType::kINT32
                    || field.length != 1 || field.data == nullptr)
                {
                    return nullptr;
                }
                valueHeads = *static_cast<int32_t const*>(field.data);
                foundValueHeads = true;
            }
        }
        if (!foundValueHeads || valueHeads < 1)
        {
            return nullptr;
        }
        auto* plugin = new (std::nothrow) AttentionValuePlugin(valueHeads);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

private:
    PluginField mAttribute{};
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrZipformerAttentionValuePlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // TensorRT maintains distinct runtime and builder registries. Registering
    // both lets packaged engines deserialize and ONNX builds discover the same
    // plugin even when those APIs use different registry instances.
    static AttentionValuePluginCreator runtimeCreator;
    static AttentionValuePluginCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry,
                                AttentionValuePluginCreator& creator) noexcept {
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
    bool const builderRegistered =
        ensureRegistered(builderRegistry, builderCreator);
    return runtimeRegistered && builderRegistered;
}
