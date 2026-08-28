// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>

#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

// TensorRT loads every plugin library with global symbol visibility. Keep all
// implementation details local to this shared object so identically named
// helpers in another plugin cannot be interposed by the dynamic linker.
namespace
{

// Fuse the masked depthwise convolution, evaluation-mode BatchNorm, and SiLU
// in the Parakeet Conformer convolution branch. Inputs and output use contiguous
// NTC storage; weights are folded by the exporter and stored as (kernel, channel).
// Valid lengths zero the padded input suffix before convolution, matching the
// PyTorch masked-fill behavior; output frames are intentionally not re-masked.
constexpr char const* kPluginName = "parakeet_conformer_convolution";
constexpr char const* kPluginVersion = "1";
constexpr int32_t kInputCount = 4;
constexpr int32_t kOutputCount = 1;
// Grid-stride loops cover larger tensors; this cap supplies enough resident
// blocks to saturate supported GPUs without creating an excessive launch grid.
constexpr int64_t kMaxBlocks = 65535;
// Keep the current adjacent-frame tactic ID stable for serialized engines;
// the ID is opaque to TensorRT and independent of the CUDA block size.
constexpr int32_t kTactic = 2304;
constexpr int32_t kThreads = 256;
constexpr std::array<int32_t, 3> kDataInputIndexes{0, 2, 3};

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

constexpr int32_t vectorWidth(DataType type) noexcept
{
    return type == DataType::kFLOAT ? 4 : 2;
}

bool hasAddressableBytes(Dims const& dims, int32_t elementBytes) noexcept
{
    if (dims.nbDims < 1 || elementBytes < 1)
    {
        return false;
    }

    int64_t elements = 1;
    int64_t const maxElements = std::numeric_limits<int64_t>::max() / elementBytes;
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

bool haveValidShapes(Dims const* inputs, Dims const& output,
    DataType type) noexcept
{
    int32_t const elementBytes = dataTypeBytes(type);
    if (inputs == nullptr || inputs[0].nbDims != 3 || inputs[1].nbDims != 1
        || inputs[2].nbDims != 2 || inputs[3].nbDims != 1
        || output.nbDims != 3 || !hasAddressableBytes(inputs[0], elementBytes)
        || !hasAddressableBytes(inputs[1], sizeof(int32_t))
        || !hasAddressableBytes(inputs[2], elementBytes)
        || !hasAddressableBytes(inputs[3], elementBytes)
        || !hasAddressableBytes(output, elementBytes))
    {
        return false;
    }

    int32_t const channels = inputs[0].d[2];
    int32_t const kernelSize = inputs[2].d[0];
    return inputs[1].d[0] == inputs[0].d[0] && kernelSize % 2 == 1
        && inputs[2].d[1] == channels && inputs[3].d[0] == channels
        && channels % vectorWidth(type) == 0
        && output.d[0] == inputs[0].d[0]
        && output.d[1] == inputs[0].d[1]
        && output.d[2] == channels;
}

// One thread evaluates one packed channel vector at one (batch, frame)
// position. Consecutive threads therefore access consecutive channels, while
// the grid-stride loop covers tensors larger than CUDA's one-dimensional grid.
// The exporter folds evaluation-mode BatchNorm into weight and bias, so each
// kernel only has to apply masked depthwise convolution followed by SiLU.
__global__ void parakeetConformerConvolutionFloat4(
    float const* __restrict__ x, int32_t const* __restrict__ validLengths,
    float const* __restrict__ weight, float const* __restrict__ bias,
    int64_t numElements, int32_t sequenceLength, int32_t numChannels,
    int32_t kernelSize, float* __restrict__ output)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int32_t const channelGroups = numChannels / 4;
        int32_t const channel = static_cast<int32_t>(index % channelGroups) * 4;
        int64_t const frameIndex = index / channelGroups;
        int32_t const frame = static_cast<int32_t>(frameIndex % sequenceLength);
        int32_t const batch = static_cast<int32_t>(frameIndex / sequenceLength);
        int32_t const requestedLength = validLengths[batch];
        int32_t const validLength = requestedLength < 0
            ? 0 : (requestedLength < sequenceLength ? requestedLength : sequenceLength);
        int32_t const padding = kernelSize / 2;
        float4 value = *reinterpret_cast<float4 const*>(bias + channel);
        for (int32_t kernel = 0; kernel < kernelSize; ++kernel)
        {
            int64_t const inputFrame = static_cast<int64_t>(frame) + kernel - padding;
            if (inputFrame >= 0 && inputFrame < validLength)
            {
                int64_t const inputIndex =
                    (static_cast<int64_t>(batch) * sequenceLength + inputFrame)
                    * numChannels + channel;
                float4 const inputValue =
                    *reinterpret_cast<float4 const*>(x + inputIndex);
                float4 const weightValue = *reinterpret_cast<float4 const*>(
                    weight + static_cast<int64_t>(kernel) * numChannels + channel);
                value.x = fmaf(inputValue.x, weightValue.x, value.x);
                value.y = fmaf(inputValue.y, weightValue.y, value.y);
                value.z = fmaf(inputValue.z, weightValue.z, value.z);
                value.w = fmaf(inputValue.w, weightValue.w, value.w);
            }
        }
        // The convolution accumulates in its storage dtype to preserve the
        // TensorRT engine's selected precision. SiLU is evaluated in float for
        // the packed low-precision paths below before converting once at store.
        value.x /= 1.0F + __expf(-value.x);
        value.y /= 1.0F + __expf(-value.y);
        value.z /= 1.0F + __expf(-value.z);
        value.w /= 1.0F + __expf(-value.w);
        *reinterpret_cast<float4*>(output + frameIndex * numChannels + channel) = value;
    }
}

template <typename T>
struct LowPrecisionOps;

template <>
struct LowPrecisionOps<half>
{
    using Pair = half2;

    static __device__ __forceinline__ Pair fma(
        Pair input, Pair weight, Pair value)
    {
        return __hfma2(input, weight, value);
    }

    static __device__ __forceinline__ float2 toFloat(Pair value)
    {
        return __half22float2(value);
    }

    static __device__ __forceinline__ Pair fromFloat(float x, float y)
    {
        return __floats2half2_rn(x, y);
    }
};

template <>
struct LowPrecisionOps<__nv_bfloat16>
{
    using Pair = __nv_bfloat162;

    static __device__ __forceinline__ Pair fma(
        Pair input, Pair weight, Pair value)
    {
        return __hfma2(input, weight, value);
    }

    static __device__ __forceinline__ float2 toFloat(Pair value)
    {
        return __bfloat1622float2(value);
    }

    static __device__ __forceinline__ Pair fromFloat(float x, float y)
    {
        return __floats2bfloat162_rn(x, y);
    }
};

template <typename T>
__device__ __forceinline__ typename LowPrecisionOps<T>::Pair siluPair(
    typename LowPrecisionOps<T>::Pair value)
{
    using Ops = LowPrecisionOps<T>;
    float2 const converted = Ops::toFloat(value);
    return Ops::fromFloat(
        converted.x / (1.0F + __expf(-converted.x)),
        converted.y / (1.0F + __expf(-converted.y)));
}

template <typename T>
struct alignas(16) PackedEightChannels
{
    typename LowPrecisionOps<T>::Pair values[4];
};

// The adjacent-frame kernel relies on one aligned 16-byte transaction for
// every eight-channel load and store.
static_assert(sizeof(PackedEightChannels<half>) == 16);
static_assert(sizeof(PackedEightChannels<__nv_bfloat16>) == 16);
static_assert(alignof(PackedEightChannels<half>) == 16);
static_assert(alignof(PackedEightChannels<__nv_bfloat16>) == 16);

// FP16 and BF16 expose matching packed-pair intrinsics. Templating these
// kernels keeps their indexing and boundary behavior identical while each
// specialization still compiles to its native storage and conversion ops.
template <typename T>
__global__ void parakeetConformerConvolutionPair(
    T const* __restrict__ x, int32_t const* __restrict__ validLengths,
    T const* __restrict__ weight, T const* __restrict__ bias,
    int64_t numElements, int32_t sequenceLength, int32_t numChannels,
    int32_t kernelSize, T* __restrict__ output)
{
    using Ops = LowPrecisionOps<T>;
    using Pair = typename Ops::Pair;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int32_t const channelPairs = numChannels / 2;
        int32_t const channel = static_cast<int32_t>(index % channelPairs) * 2;
        int64_t const frameIndex = index / channelPairs;
        int32_t const frame = static_cast<int32_t>(frameIndex % sequenceLength);
        int32_t const batch = static_cast<int32_t>(frameIndex / sequenceLength);
        int32_t const requestedLength = validLengths[batch];
        int32_t const validLength = requestedLength < 0
            ? 0
            : (requestedLength < sequenceLength ? requestedLength : sequenceLength);
        int32_t const padding = kernelSize / 2;
        Pair value = *reinterpret_cast<Pair const*>(bias + channel);
        for (int32_t kernel = 0; kernel < kernelSize; ++kernel)
        {
            int64_t const inputFrame =
                static_cast<int64_t>(frame) + kernel - padding;
            if (inputFrame >= 0 && inputFrame < validLength)
            {
                int64_t const inputIndex =
                    (static_cast<int64_t>(batch) * sequenceLength + inputFrame)
                    * numChannels + channel;
                Pair const inputValue = *reinterpret_cast<Pair const*>(x + inputIndex);
                Pair const weightValue = *reinterpret_cast<Pair const*>(
                    weight + static_cast<int64_t>(kernel) * numChannels + channel);
                value = Ops::fma(inputValue, weightValue, value);
            }
        }
        *reinterpret_cast<Pair*>(output + frameIndex * numChannels + channel)
            = siluPair<T>(value);
    }
}

// Adjacent output frames share all but one input in their receptive fields.
// Carrying that overlapping input through the tap loop reduces input reads
// from 2K to K+1 and reuses each depthwise weight for both outputs without a
// shared-memory staging barrier. One thread handles an eight-channel vector for
// one frame pair; for odd sequence lengths the unused second result is not stored.
template <typename T>
__global__ void parakeetConformerConvolutionAdjacentFrames(
    T const* __restrict__ x, int32_t const* __restrict__ validLengths,
    T const* __restrict__ weight, T const* __restrict__ bias,
    int64_t numElements, int32_t sequenceLength, int32_t numChannels,
    int32_t kernelSize, T* __restrict__ output)
{
    using Ops = LowPrecisionOps<T>;
    using Packed = PackedEightChannels<T>;
    int32_t const channelGroups = numChannels / 8;
    int32_t const framePairs =
        sequenceLength / 2 + sequenceLength % 2;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int32_t const channel = static_cast<int32_t>(index % channelGroups) * 8;
        int64_t const framePairIndex = index / channelGroups;
        int32_t const frame =
            static_cast<int32_t>(framePairIndex % framePairs) * 2;
        int32_t const batch =
            static_cast<int32_t>(framePairIndex / framePairs);
        int32_t const requestedLength = validLengths[batch];
        int32_t const validLength = requestedLength < 0
            ? 0
            : (requestedLength < sequenceLength ? requestedLength : sequenceLength);
        int32_t const padding = kernelSize / 2;
        Packed value0 = *reinterpret_cast<Packed const*>(bias + channel);
        Packed value1 = value0;

        int64_t inputFrame = static_cast<int64_t>(frame) - padding;
        Packed inputValue0{};
        if (inputFrame >= 0 && inputFrame < validLength)
        {
            int64_t const inputIndex =
                (static_cast<int64_t>(batch) * sequenceLength + inputFrame)
                    * numChannels
                + channel;
            inputValue0 = *reinterpret_cast<Packed const*>(x + inputIndex);
        }
        for (int32_t kernel = 0; kernel < kernelSize; ++kernel)
        {
            ++inputFrame;
            Packed inputValue1{};
            if (inputFrame >= 0 && inputFrame < validLength)
            {
                int64_t const inputIndex =
                    (static_cast<int64_t>(batch) * sequenceLength + inputFrame)
                        * numChannels
                    + channel;
                inputValue1 = *reinterpret_cast<Packed const*>(x + inputIndex);
            }
            Packed const weightValue = *reinterpret_cast<Packed const*>(
                weight + static_cast<int64_t>(kernel) * numChannels + channel);
#pragma unroll
            for (int32_t pair = 0; pair < 4; ++pair)
            {
                value0.values[pair] = Ops::fma(
                    inputValue0.values[pair], weightValue.values[pair],
                    value0.values[pair]);
                value1.values[pair] = Ops::fma(
                    inputValue1.values[pair], weightValue.values[pair],
                    value1.values[pair]);
            }
            inputValue0 = inputValue1;
        }

        Packed result0;
        Packed result1;
#pragma unroll
        for (int32_t pair = 0; pair < 4; ++pair)
        {
            result0.values[pair] = siluPair<T>(value0.values[pair]);
            result1.values[pair] = siluPair<T>(value1.values[pair]);
        }
        int64_t const outputIndex =
            (static_cast<int64_t>(batch) * sequenceLength + frame) * numChannels
            + channel;
        *reinterpret_cast<Packed*>(output + outputIndex) = result0;
        if (frame + 1 < sequenceLength)
        {
            *reinterpret_cast<Packed*>(output + outputIndex + numChannels) = result1;
        }
    }
}

template <typename T>
void launchLowPrecisionConvolution(T const* x,
    int32_t const* validLengths, T const* weight, T const* bias,
    T* output, bool useAdjacentFrameKernel,
    int32_t blocks, int32_t threads, int64_t numElements,
    int32_t sequenceLength, int32_t numChannels, int32_t kernelSize,
    cudaStream_t stream)
{
    if (useAdjacentFrameKernel)
    {
        parakeetConformerConvolutionAdjacentFrames<T>
            <<<blocks, threads, 0, stream>>>(x, validLengths, weight, bias,
                numElements, sequenceLength, numChannels, kernelSize, output);
    }
    else
    {
        parakeetConformerConvolutionPair<T>
            <<<blocks, threads, 0, stream>>>(x, validLengths, weight, bias,
                numElements, sequenceLength, numChannels, kernelSize, output);
    }
}

// Current builds use the measured 256-thread adjacent-frame tactic for aligned
// low-precision tensors. The pair kernel preserves support for channel counts
// that cannot use the eight-channel vectorization, while FP32 uses float4.
class ParakeetConvolutionPlugin final : public IPluginV3, public IPluginV3OneCore,
                                        public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
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
        return new (std::nothrow) ParakeetConvolutionPlugin();
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
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || inputs[0].desc.dims.nbDims != 3
            || inputs[1].desc.dims.nbDims != 1
            || inputs[2].desc.dims.nbDims != 2
            || inputs[3].desc.dims.nbDims != 1
            || outputs[0].desc.dims.nbDims != 3
            || !isSupportedDataType(inputs[0].desc.type)
            || inputs[1].desc.type != DataType::kINT32
            || outputs[0].desc.type != inputs[0].desc.type
            || outputs[0].desc.format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index : kDataInputIndexes)
        {
            if (inputs[index].desc.type != inputs[0].desc.type
                || inputs[index].desc.format != TensorFormat::kLINEAR)
            {
                return 1;
            }
        }
        if (inputs[1].desc.format != TensorFormat::kLINEAR)
        {
            return 1;
        }

        Dims minInputs[kInputCount]{};
        Dims optInputs[kInputCount]{};
        Dims maxInputs[kInputCount]{};
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            minInputs[index] = inputs[index].min;
            optInputs[index] = inputs[index].opt;
            maxInputs[index] = inputs[index].max;
        }
        return haveValidShapes(minInputs, outputs[0].min, inputs[0].desc.type)
                && haveValidShapes(
                    optInputs, outputs[0].opt, inputs[0].desc.type)
                && haveValidShapes(
                    maxInputs, outputs[0].max, inputs[0].desc.type)
            ? 0
            : 1;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbInputs != kInputCount || nbOutputs != kOutputCount
            || !isSupportedDataType(inputTypes[0])
            || inputTypes[1] != DataType::kINT32
            || inputTypes[2] != inputTypes[0]
            || inputTypes[3] != inputTypes[0])
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder&) noexcept override
    {
        static_cast<void>(shapeInputs);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || nbShapeInputs != 0
            || inputs[0].nbDims != 3 || inputs[0].d[0] == nullptr
            || inputs[0].d[1] == nullptr || inputs[0].d[2] == nullptr
            || inputs[1].nbDims != 1 || inputs[2].nbDims != 2
            || inputs[3].nbDims != 1)
        {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
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
        if (pos == 1)
        {
            return desc.type == DataType::kINT32;
        }
        return isSupportedDataType(desc.type)
            && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t getNbTactics() noexcept override
    {
        return 1;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        if (tactics == nullptr || nbTactics != 1)
        {
            return 1;
        }
        tactics[0] = kTactic;
        return 0;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        return tactic == kTactic ? 0 : 1;
    }

    char const* getTimingCacheID() noexcept override
    {
        // The plugin has no configurable fields. TensorRT still includes the
        // concrete input profile in its timing key, so equivalent Conformer
        // layers can share results without conflating different shapes.
        return kPluginName;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || !isSupportedDataType(inputs[0].type)
            || inputs[1].type != DataType::kINT32
            || outputs[0].type != inputs[0].type
            || outputs[0].format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index : kDataInputIndexes)
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
        return inputs[1].format == TensorFormat::kLINEAR
                && haveValidShapes(
                    inputShapes, outputs[0].dims, inputs[0].type)
            ? 0
            : 1;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        static_cast<void>(workspace);
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || outputs[0] == nullptr)
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

        // TensorRT calls onShapeChange before enqueue. Rechecking here keeps the
        // memory-safety contract local to the launch when a runtime invokes the
        // plugin outside the usual execution-context lifecycle.
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

        int32_t const batch = static_cast<int32_t>(inputDesc[0].dims.d[0]);
        int32_t const sequenceLength = static_cast<int32_t>(inputDesc[0].dims.d[1]);
        int32_t const numChannels = static_cast<int32_t>(inputDesc[0].dims.d[2]);
        int32_t const kernelSize = static_cast<int32_t>(inputDesc[2].dims.d[0]);
        bool const useAdjacentFrameKernel =
            inputDesc[0].type != DataType::kFLOAT && numChannels % 8 == 0;
        int32_t const width = useAdjacentFrameKernel
            ? 8 : vectorWidth(inputDesc[0].type);
        int32_t const temporalElements = useAdjacentFrameKernel
            ? sequenceLength / 2 + sequenceLength % 2 : sequenceLength;
        int64_t const numElements = static_cast<int64_t>(batch)
            * temporalElements * (numChannels / width);
        int32_t const threads = kThreads;
        int64_t const requiredBlocks =
            numElements / threads + (numElements % threads != 0);
        int32_t const blocks = static_cast<int32_t>(
            std::min<int64_t>(kMaxBlocks, requiredBlocks));
        if (inputDesc[0].type == DataType::kHALF)
        {
            launchLowPrecisionConvolution(
                static_cast<half const*>(inputs[0]),
                static_cast<int32_t const*>(inputs[1]),
                static_cast<half const*>(inputs[2]),
                static_cast<half const*>(inputs[3]),
                static_cast<half*>(outputs[0]), useAdjacentFrameKernel,
                blocks, threads, numElements,
                sequenceLength, numChannels, kernelSize, stream);
        }
        else if (inputDesc[0].type == DataType::kBF16)
        {
            launchLowPrecisionConvolution(
                static_cast<__nv_bfloat16 const*>(inputs[0]),
                static_cast<int32_t const*>(inputs[1]),
                static_cast<__nv_bfloat16 const*>(inputs[2]),
                static_cast<__nv_bfloat16 const*>(inputs[3]),
                static_cast<__nv_bfloat16*>(outputs[0]),
                useAdjacentFrameKernel, blocks, threads,
                numElements, sequenceLength, numChannels, kernelSize, stream);
        }
        else
        {
            parakeetConformerConvolutionFloat4<<<blocks, threads, 0, stream>>>(
                static_cast<float const*>(inputs[0]),
                static_cast<int32_t const*>(inputs[1]),
                static_cast<float const*>(inputs[2]),
                static_cast<float const*>(inputs[3]), numElements,
                sequenceLength, numChannels, kernelSize,
                static_cast<float*>(outputs[0]));
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    PluginFieldCollection mFields{};
};

class ParakeetConvolutionPluginCreator final : public IPluginCreatorV3One
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
        // The plugin has no serialized attributes. Reject unexpected fields so
        // malformed ONNX nodes cannot be accepted silently.
        if (fields == nullptr || fields->nbFields != 0)
        {
            return nullptr;
        }
        return new (std::nothrow) ParakeetConvolutionPlugin();
    }

private:
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrParakeetConvolutionPlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Builder and runtime use distinct registries. Treat an existing matching
    // creator as success so repeated package imports remain idempotent.
    static ParakeetConvolutionPluginCreator runtimeCreator;
    static ParakeetConvolutionPluginCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry,
                                ParakeetConvolutionPluginCreator& creator) noexcept {
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
            || registry->getCreator(kPluginName, kPluginVersion,
                   kPluginNamespace)
                != nullptr;
    };

    bool const runtimeRegistered =
        ensureRegistered(getPluginRegistry(), runtimeCreator);
    auto* builderRegistry = nvinfer1::getBuilderPluginRegistry(
        nvinfer1::EngineCapability::kSTANDARD);
    bool const builderRegistered =
        ensureRegistered(builderRegistry, builderCreator);
    return runtimeRegistered && builderRegistered;
}
