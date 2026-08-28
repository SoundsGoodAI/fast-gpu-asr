// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <new>
#include <string_view>

#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

namespace
{

// Both operations consume and produce contiguous NTC representations.
// Downsampling repeats the final input frame to complete a partial group,
// while upsampling repeats each lower-rate frame and applies a per-channel
// bypass interpolation.
constexpr char const* kDownsampleName = "zipformer_downsample";
constexpr char const* kUpsampleName = "zipformer_upsample_bypass";
constexpr char const* kPluginVersion = "1";
constexpr char const* kFactorField = "factor";
constexpr int32_t kOutputCount = 1;
constexpr int32_t kThreadsPerBlock = 256;
constexpr int64_t kMaxGridDimensionYZ = 65535;

enum class Operation
{
    kDownsample,
    kUpsample,
};

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

constexpr int32_t expectedInputCount(Operation operation) noexcept
{
    return operation == Operation::kDownsample ? 2 : 3;
}

bool hasAddressableBytes(Dims const& dims, int32_t elementBytes) noexcept
{
    // Kernel offsets use signed 64-bit element indexes, while pointer
    // arithmetic ultimately addresses bytes. Prove both before narrowing
    // individual dimensions for launch arguments.
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

bool haveValidDownsampleShapes(Dims const& input, Dims const& weights,
    Dims const& output, int32_t factor, int32_t elementBytes) noexcept
{
    if (factor < 1 || input.nbDims != 3 || weights.nbDims != 2
        || output.nbDims != 3 || !hasAddressableBytes(input, elementBytes)
        || !hasAddressableBytes(weights, elementBytes)
        || !hasAddressableBytes(output, elementBytes))
    {
        return false;
    }

    int64_t const outputLength =
        input.d[1] / factor + (input.d[1] % factor != 0);
    return input.d[0] <= kMaxGridDimensionYZ
        && outputLength <= kMaxGridDimensionYZ
        && input.d[1] <= std::numeric_limits<int32_t>::max()
        && input.d[2] <= std::numeric_limits<int32_t>::max()
        && weights.d[0] == factor && weights.d[1] == 1
        && output.d[0] == input.d[0] && output.d[1] == outputLength
        && output.d[2] == input.d[2];
}

bool haveValidUpsampleShapes(Dims const& early, Dims const& later,
    Dims const& scale, Dims const& output, int32_t factor,
    int32_t elementBytes) noexcept
{
    if (factor < 1 || early.nbDims != 3 || later.nbDims != 3
        || scale.nbDims != 1 || output.nbDims != 3
        || !hasAddressableBytes(early, elementBytes)
        || !hasAddressableBytes(later, elementBytes)
        || !hasAddressableBytes(scale, elementBytes)
        || !hasAddressableBytes(output, elementBytes))
    {
        return false;
    }

    int64_t const laterLength =
        early.d[1] / factor + (early.d[1] % factor != 0);
    return early.d[0] <= kMaxGridDimensionYZ
        && early.d[1] <= kMaxGridDimensionYZ
        && early.d[2] <= std::numeric_limits<int32_t>::max()
        && later.d[0] == early.d[0] && later.d[1] == laterLength
        && later.d[2] == early.d[2] && scale.d[0] == early.d[2]
        && output.d[0] == early.d[0] && output.d[1] == early.d[1]
        && output.d[2] == early.d[2];
}

bool haveValidShapes(Operation operation, Dims const* inputs,
    Dims const& output, int32_t factor, DataType type) noexcept
{
    if (inputs == nullptr)
    {
        return false;
    }
    int32_t const elementBytes = dataTypeBytes(type);
    return operation == Operation::kDownsample
        ? haveValidDownsampleShapes(
              inputs[0], inputs[1], output, factor, elementBytes)
        : haveValidUpsampleShapes(
              inputs[0], inputs[1], inputs[2], output, factor, elementBytes);
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
__global__ void downsample(T const* input, T const* weights, T* output,
    int32_t inputLength, int32_t outputLength, int32_t channels,
    int32_t factor)
{
    // Grid axes map to channel, output frame, and batch. Each thread owns one
    // output scalar and accumulates its temporal group in FP32. The final
    // input frame is repeated when the sequence does not fill the last group.
    int64_t const channel =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int64_t const firstInputFrame =
        static_cast<int64_t>(outputFrame) * factor;
    float value = 0.0F;
    for (int64_t offset = 0; offset < factor; ++offset)
    {
        int64_t const candidateFrame = firstInputFrame + offset;
        int32_t const inputFrame = candidateFrame < inputLength
            ? static_cast<int32_t>(candidateFrame)
            : inputLength - 1;
        int64_t const inputIndex =
            (static_cast<int64_t>(batch) * inputLength + inputFrame) * channels
            + channel;
        value += toFloat(input[inputIndex]) * toFloat(weights[offset]);
    }
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame) * channels
        + channel;
    output[outputIndex] = fromFloat<T>(value);
}

__global__ void downsampleHalf2(half2 const* input, half const* weights,
    half2* output, int32_t inputLength, int32_t outputLength,
    int32_t channelPairs, int32_t factor)
{
    // Even channel counts permit two adjacent values to share indexing and
    // weight loads while retaining independent FP32 accumulators.
    int64_t const channelPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channelPair >= channelPairs)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int64_t const firstInputFrame =
        static_cast<int64_t>(outputFrame) * factor;
    float2 value = make_float2(0.0F, 0.0F);
    for (int64_t offset = 0; offset < factor; ++offset)
    {
        int64_t const candidateFrame = firstInputFrame + offset;
        int32_t const inputFrame = candidateFrame < inputLength
            ? static_cast<int32_t>(candidateFrame)
            : inputLength - 1;
        int64_t const inputIndex =
            (static_cast<int64_t>(batch) * inputLength + inputFrame)
                * channelPairs
            + channelPair;
        float2 const sample = __half22float2(input[inputIndex]);
        float const weight = __half2float(weights[offset]);
        value.x += sample.x * weight;
        value.y += sample.y * weight;
    }
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame)
            * channelPairs
        + channelPair;
    output[outputIndex] = __floats2half2_rn(value.x, value.y);
}

__global__ void downsampleBfloat162(__nv_bfloat162 const* input,
    __nv_bfloat16 const* weights, __nv_bfloat162* output,
    int32_t inputLength, int32_t outputLength, int32_t channelPairs,
    int32_t factor)
{
    int64_t const channelPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channelPair >= channelPairs)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int64_t const firstInputFrame =
        static_cast<int64_t>(outputFrame) * factor;
    float2 value = make_float2(0.0F, 0.0F);
    for (int64_t offset = 0; offset < factor; ++offset)
    {
        int64_t const candidateFrame = firstInputFrame + offset;
        int32_t const inputFrame = candidateFrame < inputLength
            ? static_cast<int32_t>(candidateFrame)
            : inputLength - 1;
        int64_t const inputIndex =
            (static_cast<int64_t>(batch) * inputLength + inputFrame)
                * channelPairs
            + channelPair;
        float2 const sample = __bfloat1622float2(input[inputIndex]);
        float const weight = __bfloat162float(weights[offset]);
        value.x += sample.x * weight;
        value.y += sample.y * weight;
    }
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame)
            * channelPairs
        + channelPair;
    output[outputIndex] = __floats2bfloat162_rn(value.x, value.y);
}

template <typename T>
__global__ void upsampleBypass(T const* early, T const* later,
    T const* scale, T* output, int32_t outputLength, int32_t laterLength,
    int32_t channels, int32_t factor)
{
    // Repeat the lower-rate frame selected by integer division, then apply
    // early + (later - early) * scale independently for every channel.
    int64_t const channel =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int32_t const candidateFrame = outputFrame / factor;
    int32_t const laterFrame =
        candidateFrame < laterLength ? candidateFrame : laterLength - 1;
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame) * channels
        + channel;
    int64_t const laterIndex =
        (static_cast<int64_t>(batch) * laterLength + laterFrame) * channels
        + channel;
    float const earlyValue = toFloat(early[outputIndex]);
    float const laterValue = toFloat(later[laterIndex]);
    float const channelScale = toFloat(scale[channel]);
    output[outputIndex] = fromFloat<T>(
        earlyValue + (laterValue - earlyValue) * channelScale);
}

__global__ void upsampleBypassHalf2(half2 const* early,
    half2 const* later, half2 const* scale, half2* output,
    int32_t outputLength, int32_t laterLength, int32_t channelPairs,
    int32_t factor)
{
    int64_t const channelPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channelPair >= channelPairs)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int32_t const candidateFrame = outputFrame / factor;
    int32_t const laterFrame =
        candidateFrame < laterLength ? candidateFrame : laterLength - 1;
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame)
            * channelPairs
        + channelPair;
    int64_t const laterIndex =
        (static_cast<int64_t>(batch) * laterLength + laterFrame)
            * channelPairs
        + channelPair;
    float2 const earlyValue = __half22float2(early[outputIndex]);
    float2 const laterValue = __half22float2(later[laterIndex]);
    float2 const channelScale = __half22float2(scale[channelPair]);
    output[outputIndex] = __floats2half2_rn(
        earlyValue.x
            + (laterValue.x - earlyValue.x) * channelScale.x,
        earlyValue.y
            + (laterValue.y - earlyValue.y) * channelScale.y);
}

__global__ void upsampleBypassBfloat162(__nv_bfloat162 const* early,
    __nv_bfloat162 const* later, __nv_bfloat162 const* scale,
    __nv_bfloat162* output, int32_t outputLength, int32_t laterLength,
    int32_t channelPairs, int32_t factor)
{
    int64_t const channelPair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channelPair >= channelPairs)
    {
        return;
    }

    int32_t const outputFrame = blockIdx.y;
    int32_t const batch = blockIdx.z;
    int32_t const candidateFrame = outputFrame / factor;
    int32_t const laterFrame =
        candidateFrame < laterLength ? candidateFrame : laterLength - 1;
    int64_t const outputIndex =
        (static_cast<int64_t>(batch) * outputLength + outputFrame)
            * channelPairs
        + channelPair;
    int64_t const laterIndex =
        (static_cast<int64_t>(batch) * laterLength + laterFrame)
            * channelPairs
        + channelPair;
    float2 const earlyValue = __bfloat1622float2(early[outputIndex]);
    float2 const laterValue = __bfloat1622float2(later[laterIndex]);
    float2 const channelScale = __bfloat1622float2(scale[channelPair]);
    output[outputIndex] = __floats2bfloat162_rn(
        earlyValue.x
            + (laterValue.x - earlyValue.x) * channelScale.x,
        earlyValue.y
            + (laterValue.y - earlyValue.y) * channelScale.y);
}

class ResamplingPlugin final : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime
{
public:
    ResamplingPlugin(Operation operation, int32_t factor) noexcept
        : mOperation(operation), mFactor(factor)
    {
        mSerializedField = {
            kFactorField, &mFactor, PluginFieldType::kINT32, 1};
        mFields = {1, &mSerializedField};
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
        if (mFactor < 1)
        {
            return nullptr;
        }
        return new (std::nothrow) ResamplingPlugin(mOperation, mFactor);
    }

    char const* getPluginName() const noexcept override
    {
        return mOperation == Operation::kDownsample ? kDownsampleName
                                                    : kUpsampleName;
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
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (inputs == nullptr || outputs == nullptr
            || nbInputs != expectedInputs || nbOutputs != kOutputCount
            || mFactor < 1
            || inputs[0].desc.dims.nbDims != 3
            || outputs[0].desc.dims.nbDims != 3
            || (mOperation == Operation::kDownsample
                && inputs[1].desc.dims.nbDims != 2)
            || (mOperation == Operation::kUpsample
                && (inputs[1].desc.dims.nbDims != 3
                    || inputs[2].desc.dims.nbDims != 1))
            || !isSupportedDataType(inputs[0].desc.type)
            || outputs[0].desc.type != inputs[0].desc.type
            || inputs[0].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index = 1; index < expectedInputs; ++index)
        {
            if (inputs[index].desc.type != inputs[0].desc.type
                || inputs[index].desc.format != TensorFormat::kLINEAR)
            {
                return 1;
            }
        }

        Dims minInputs[3]{};
        Dims optInputs[3]{};
        Dims maxInputs[3]{};
        for (int32_t index = 0; index < expectedInputs; ++index)
        {
            minInputs[index] = inputs[index].min;
            optInputs[index] = inputs[index].opt;
            maxInputs[index] = inputs[index].max;
        }
        return haveValidShapes(
                   mOperation, minInputs, outputs[0].min, mFactor,
                   inputs[0].desc.type)
                && haveValidShapes(
                    mOperation, optInputs, outputs[0].opt, mFactor,
                    inputs[0].desc.type)
                && haveValidShapes(
                    mOperation, maxInputs, outputs[0].max, mFactor,
                    inputs[0].desc.type)
            ? 0
            : 1;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbInputs != expectedInputs || nbOutputs != kOutputCount
            || !isSupportedDataType(inputTypes[0]))
        {
            return 1;
        }
        for (int32_t index = 1; index < expectedInputs; ++index)
        {
            if (inputTypes[index] != inputTypes[0])
            {
                return 1;
            }
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
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (inputs == nullptr || outputs == nullptr
            || nbInputs != expectedInputs || nbOutputs != kOutputCount
            || nbShapeInputs != 0 || mFactor < 1
            || inputs[0].nbDims != 3 || inputs[0].d[0] == nullptr
            || inputs[0].d[1] == nullptr || inputs[0].d[2] == nullptr
            || (mOperation == Operation::kDownsample
                && inputs[1].nbDims != 2)
            || (mOperation == Operation::kUpsample
                && (inputs[1].nbDims != 3 || inputs[2].nbDims != 1)))
        {
            return 1;
        }

        outputs[0] = inputs[0];
        if (mOperation == Operation::kDownsample)
        {
            auto* const one = exprBuilder.constant(1);
            auto* const divisor = exprBuilder.constant(mFactor);
            if (one == nullptr || divisor == nullptr)
            {
                return 1;
            }
            auto* const adjustedLength = exprBuilder.operation(
                DimensionOperation::kSUB, *inputs[0].d[1], *one);
            if (adjustedLength == nullptr)
            {
                return 1;
            }
            auto* const quotient = exprBuilder.operation(
                DimensionOperation::kFLOOR_DIV, *adjustedLength, *divisor);
            if (quotient == nullptr)
            {
                return 1;
            }
            auto* const outputLength = exprBuilder.operation(
                DimensionOperation::kSUM, *quotient, *one);
            if (outputLength == nullptr)
            {
                return 1;
            }
            outputs[0].d[1] = outputLength;
        }
        return 0;
    }

    bool supportsFormatCombination(int32_t pos,
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (inOut == nullptr || nbInputs != expectedInputs
            || nbOutputs != kOutputCount || pos < 0
            || pos >= expectedInputs + kOutputCount)
        {
            return false;
        }
        auto const& desc = inOut[pos].desc;
        return desc.format == TensorFormat::kLINEAR
            && isSupportedDataType(desc.type)
            && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        static_cast<void>(inputs);
        static_cast<void>(nbInputs);
        static_cast<void>(outputs);
        static_cast<void>(nbOutputs);
        return 0;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (inputs == nullptr || outputs == nullptr
            || nbInputs != expectedInputs || nbOutputs != kOutputCount
            || !isSupportedDataType(inputs[0].type)
            || outputs[0].type != inputs[0].type
            || inputs[0].format != TensorFormat::kLINEAR
            || outputs[0].format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index = 1; index < expectedInputs; ++index)
        {
            if (inputs[index].type != inputs[0].type
                || inputs[index].format != TensorFormat::kLINEAR)
            {
                return 1;
            }
        }

        Dims inputShapes[3]{};
        for (int32_t index = 0; index < expectedInputs; ++index)
        {
            inputShapes[index] = inputs[index].dims;
        }
        return haveValidShapes(
                   mOperation, inputShapes, outputs[0].dims, mFactor,
                   inputs[0].type)
            ? 0
            : 1;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        static_cast<void>(workspace);
        int32_t const expectedInputs = expectedInputCount(mOperation);
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || outputs[0] == nullptr)
        {
            return 1;
        }
        for (int32_t index = 0; index < expectedInputs; ++index)
        {
            if (inputs[index] == nullptr)
            {
                return 1;
            }
        }
        if (onShapeChange(
                inputDesc, expectedInputs, outputDesc, kOutputCount)
            != 0)
        {
            return 1;
        }

        // Do not consume an error left by an earlier asynchronous launch. A
        // pre-existing error means this invocation cannot report success
        // reliably, while cudaPeekAtLastError preserves it for its owner.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        int32_t const batch = static_cast<int32_t>(inputDesc[0].dims.d[0]);
        int32_t const inputLength =
            static_cast<int32_t>(inputDesc[0].dims.d[1]);
        int32_t const channels =
            static_cast<int32_t>(inputDesc[0].dims.d[2]);
        int32_t const outputLength =
            static_cast<int32_t>(outputDesc[0].dims.d[1]);
        dim3 const block(kThreadsPerBlock);

        // TensorRT allocations satisfy vector alignment. An even channel
        // stride keeps every NTC row aligned, so FP16/BF16 can process two
        // adjacent channels per thread while accumulating in FP32.
        if ((inputDesc[0].type == DataType::kHALF
                || inputDesc[0].type == DataType::kBF16)
            && channels % 2 == 0)
        {
            int32_t const channelPairs = channels / 2;
            uint32_t const channelBlocks = static_cast<uint32_t>(
                (static_cast<int64_t>(channelPairs) + kThreadsPerBlock - 1)
                / kThreadsPerBlock);
            dim3 const grid(channelBlocks, outputLength, batch);
            if (inputDesc[0].type == DataType::kHALF)
            {
                if (mOperation == Operation::kDownsample)
                {
                    downsampleHalf2<<<grid, block, 0, stream>>>(
                        static_cast<half2 const*>(inputs[0]),
                        static_cast<half const*>(inputs[1]),
                        static_cast<half2*>(outputs[0]), inputLength,
                        outputLength, channelPairs, mFactor);
                }
                else
                {
                    upsampleBypassHalf2<<<grid, block, 0, stream>>>(
                        static_cast<half2 const*>(inputs[0]),
                        static_cast<half2 const*>(inputs[1]),
                        static_cast<half2 const*>(inputs[2]),
                        static_cast<half2*>(outputs[0]), outputLength,
                        static_cast<int32_t>(inputDesc[1].dims.d[1]),
                        channelPairs, mFactor);
                }
            }
            else if (mOperation == Operation::kDownsample)
            {
                downsampleBfloat162<<<grid, block, 0, stream>>>(
                    static_cast<__nv_bfloat162 const*>(inputs[0]),
                    static_cast<__nv_bfloat16 const*>(inputs[1]),
                    static_cast<__nv_bfloat162*>(outputs[0]), inputLength,
                    outputLength, channelPairs, mFactor);
            }
            else
            {
                upsampleBypassBfloat162<<<grid, block, 0, stream>>>(
                    static_cast<__nv_bfloat162 const*>(inputs[0]),
                    static_cast<__nv_bfloat162 const*>(inputs[1]),
                    static_cast<__nv_bfloat162 const*>(inputs[2]),
                    static_cast<__nv_bfloat162*>(outputs[0]), outputLength,
                    static_cast<int32_t>(inputDesc[1].dims.d[1]),
                    channelPairs, mFactor);
            }
        }
        else
        {
            uint32_t const channelBlocks = static_cast<uint32_t>(
                (static_cast<int64_t>(channels) + kThreadsPerBlock - 1)
                / kThreadsPerBlock);
            dim3 const grid(channelBlocks, outputLength, batch);
            switch (inputDesc[0].type)
            {
            case DataType::kHALF:
                if (mOperation == Operation::kDownsample)
                {
                    downsample<<<grid, block, 0, stream>>>(
                        static_cast<half const*>(inputs[0]),
                        static_cast<half const*>(inputs[1]),
                        static_cast<half*>(outputs[0]), inputLength,
                        outputLength, channels, mFactor);
                }
                else
                {
                    upsampleBypass<<<grid, block, 0, stream>>>(
                        static_cast<half const*>(inputs[0]),
                        static_cast<half const*>(inputs[1]),
                        static_cast<half const*>(inputs[2]),
                        static_cast<half*>(outputs[0]), outputLength,
                        static_cast<int32_t>(inputDesc[1].dims.d[1]), channels,
                        mFactor);
                }
                break;
            case DataType::kBF16:
                if (mOperation == Operation::kDownsample)
                {
                    downsample<<<grid, block, 0, stream>>>(
                        static_cast<__nv_bfloat16 const*>(inputs[0]),
                        static_cast<__nv_bfloat16 const*>(inputs[1]),
                        static_cast<__nv_bfloat16*>(outputs[0]), inputLength,
                        outputLength, channels, mFactor);
                }
                else
                {
                    upsampleBypass<<<grid, block, 0, stream>>>(
                        static_cast<__nv_bfloat16 const*>(inputs[0]),
                        static_cast<__nv_bfloat16 const*>(inputs[1]),
                        static_cast<__nv_bfloat16 const*>(inputs[2]),
                        static_cast<__nv_bfloat16*>(outputs[0]), outputLength,
                        static_cast<int32_t>(inputDesc[1].dims.d[1]), channels,
                        mFactor);
                }
                break;
            case DataType::kFLOAT:
                if (mOperation == Operation::kDownsample)
                {
                    downsample<<<grid, block, 0, stream>>>(
                        static_cast<float const*>(inputs[0]),
                        static_cast<float const*>(inputs[1]),
                        static_cast<float*>(outputs[0]), inputLength,
                        outputLength, channels, mFactor);
                }
                else
                {
                    upsampleBypass<<<grid, block, 0, stream>>>(
                        static_cast<float const*>(inputs[0]),
                        static_cast<float const*>(inputs[1]),
                        static_cast<float const*>(inputs[2]),
                        static_cast<float*>(outputs[0]), outputLength,
                        static_cast<int32_t>(inputDesc[1].dims.d[1]), channels,
                        mFactor);
                }
                break;
            default: return 1;
            }
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(
        IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    Operation mOperation;
    int32_t mFactor;
    PluginField mSerializedField{};
    PluginFieldCollection mFields{};
};

class ResamplingCreator final : public IPluginCreatorV3One
{
public:
    explicit ResamplingCreator(Operation operation) noexcept
        : mOperation(operation)
    {
        mAttribute = {kFactorField, nullptr, PluginFieldType::kINT32, 1};
        mFields = {1, &mAttribute};
    }

    char const* getPluginName() const noexcept override
    {
        return mOperation == Operation::kDownsample ? kDownsampleName
                                                    : kUpsampleName;
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
        if (fields == nullptr || fields->nbFields < 0
            || (fields->nbFields > 0 && fields->fields == nullptr))
        {
            return nullptr;
        }

        int32_t factor = 0;
        bool foundFactor = false;
        // The factor is required serialized state. Reject duplicate or
        // malformed attributes rather than dereferencing parser-owned data.
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name != nullptr
                && std::string_view(field.name) == kFactorField)
            {
                if (foundFactor || field.type != PluginFieldType::kINT32
                    || field.length != 1 || field.data == nullptr)
                {
                    return nullptr;
                }
                factor = *static_cast<int32_t const*>(field.data);
                foundFactor = true;
            }
        }
        if (!foundFactor || factor < 1)
        {
            return nullptr;
        }
        return new (std::nothrow) ResamplingPlugin(mOperation, factor);
    }

private:
    Operation mOperation;
    PluginField mAttribute{};
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrZipformerResamplingPlugins() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Builder and runtime use distinct registries. Treat an existing matching
    // creator as success so repeated package imports remain idempotent.
    static ResamplingCreator runtimeDownsample(Operation::kDownsample);
    static ResamplingCreator runtimeUpsample(Operation::kUpsample);
    static ResamplingCreator builderDownsample(Operation::kDownsample);
    static ResamplingCreator builderUpsample(Operation::kUpsample);
    auto ensureRegistered = [](IPluginRegistry* registry,
                                ResamplingCreator& creator) noexcept {
        if (registry == nullptr)
        {
            return false;
        }
        if (registry->getCreator(creator.getPluginName(), kPluginVersion,
                kPluginNamespace)
            != nullptr)
        {
            return true;
        }
        return registry->registerCreator(creator, kPluginNamespace)
            || registry->getCreator(creator.getPluginName(), kPluginVersion,
                   kPluginNamespace)
                != nullptr;
    };

    auto* runtimeRegistry = getPluginRegistry();
    bool const runtimeRegistered =
        ensureRegistered(runtimeRegistry, runtimeDownsample)
        && ensureRegistered(runtimeRegistry, runtimeUpsample);
    auto* builderRegistry =
        nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    bool const builderRegistered =
        ensureRegistered(builderRegistry, builderDownsample)
        && ensureRegistered(builderRegistry, builderUpsample);
    return runtimeRegistered && builderRegistered;
}
