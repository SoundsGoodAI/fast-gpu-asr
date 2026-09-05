// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
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

namespace
{

constexpr char const* kPluginName = "zipformer_feature_extractor";
constexpr char const* kPluginVersion = "1";
constexpr int32_t kInputCount = 4;
constexpr int32_t kOutputCount = 2;
constexpr char const* kFrameLengthField = "frame_length";
constexpr char const* kFrameShiftField = "frame_shift";
constexpr char const* kLeftPaddingField = "left_padding";
constexpr char const* kMinFramesField = "min_frames";
constexpr char const* kPreemphField = "preemph";
constexpr char const* kZeroLogField = "zero_log";
constexpr size_t kCublasWorkspaceBytes = 16U << 20;
constexpr size_t kCublasWorkspaceAlignment = 256;
constexpr size_t kMaximumWorkspaceBytes =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int32_t kMaxBlocks = 65535;
constexpr size_t kTimingCacheIdSize = 256;
constexpr size_t kPortableSharedMemoryBytes = 48U << 10;
constexpr int32_t kMaxFrameLength = static_cast<int32_t>(
    (kPortableSharedMemoryBytes - kWarpsPerBlock * sizeof(float)) / sizeof(float));
constexpr int32_t kMaxFrequencies =
    static_cast<int32_t>(kPortableSharedMemoryBytes / sizeof(cufftComplex));
// FAST_16F converts the unbounded FP32 power spectrum to FP16 before the mel
// GEMM. Full-scale PCM can exceed FP16's finite range, so retain only compute
// modes with an FP32-like exponent range for this projection.
constexpr std::array<int32_t, 3> kMelProjectionTactics{
    kStrictComputeTactic,
    kFast16BFComputeTactic,
    kFastTF32ComputeTactic,
};
static_assert(kThreadsPerBlock % kWarpSize == 0);

bool isMelProjectionTactic(int32_t tactic) noexcept
{
    return std::find(kMelProjectionTactics.begin(), kMelProjectionTactics.end(), tactic)
        != kMelProjectionTactics.end();
}

struct WorkspaceLayout
{
    size_t cublasOffset{};
    size_t totalBytes{};
};

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
    size_t padded{};
    if (alignment == 0 || !checkedAdd(value, alignment - 1, padded))
    {
        return false;
    }
    output = padded / alignment * alignment;
    return true;
}

bool hasAddressableBytes(Dims const& dims, size_t elementBytes) noexcept
{
    if (dims.nbDims < 1 || elementBytes < 1)
    {
        return false;
    }
    int64_t elements = 1;
    int64_t const maxElements =
        std::numeric_limits<int64_t>::max() / static_cast<int64_t>(elementBytes);
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

bool makeWorkspaceLayout(
    int64_t rows, int32_t fftLength, int32_t frequencies, WorkspaceLayout& layout) noexcept
{
    if (rows < 1 || fftLength < 1 || frequencies < 1
        || fftLength > std::numeric_limits<int32_t>::max() - 2
        || 2LL * (frequencies - 1) != fftLength)
    {
        return false;
    }

    size_t transformElements{};
    size_t transformBytes{};
    // Workspace layout:
    // [rows x (fftLength + 2) float frames / spectrum / power]
    // [alignment padding]
    // [cuBLAS scratch]
    //
    // cuFFT's in-place R2C layout stores fftLength / 2 + 1 complex values in
    // fftLength + 2 real slots. After the transform, each row stages its complex
    // bins before overwriting the beginning of that same row with power values.
    int32_t const transformStride = fftLength + 2;
    if (!checkedMultiply(
            static_cast<size_t>(rows), static_cast<size_t>(transformStride), transformElements)
        || !checkedMultiply(transformElements, sizeof(float), transformBytes)
        || !checkedAlignUp(transformBytes, kCublasWorkspaceAlignment, layout.cublasOffset)
        || !checkedAdd(layout.cublasOffset, kCublasWorkspaceBytes, layout.totalBytes))
    {
        return false;
    }
    // TensorRT's surrounding ForeignNode path uses signed 32-bit workspace byte
    // offsets on supported releases. Reject oversized profiles during build
    // instead of allowing an internal offset to wrap during tactic execution.
    return layout.totalBytes <= kMaximumWorkspaceBytes;
}

__global__ void prepareFrames(float const* audio, int64_t const* audioLengths,
    float const* window, float* frames, int32_t* featureLengths, int32_t audioSamples,
    int32_t numFrames, int32_t frameLength, int32_t frameShift, int32_t leftPadding,
    int32_t fftLength, int32_t minFrames, float preemph)
{
    // One block owns one frame. Threads cooperatively reflect the left edge,
    // stage samples, and compute the frame mean before applying pre-emphasis and
    // the caller-provided Povey window into a zero-padded cuFFT row.
    int32_t const flattenedFrame = static_cast<int32_t>(blockIdx.x);
    int32_t const batch = flattenedFrame / numFrames;
    int32_t const frame = flattenedFrame - batch * numFrames;
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    if (frame == 0 && thread == 0)
    {
        // Exactly one block per utterance owns frame zero, avoiding a separate
        // feature-length kernel and any cross-block synchronization.
        int64_t validSamples = audioLengths[batch];
        validSamples = validSamples < 0
            ? 0 : (validSamples > audioSamples ? audioSamples : validSamples);
        int64_t const roundedLength = (validSamples + frameShift / 2) / frameShift;
        // Runtime sample counts live in device memory and therefore cannot be
        // validated while TensorRT builds the engine. Bound the published length
        // to the physical output so malformed values cannot make a downstream
        // layer read beyond the feature tensor.
        int64_t const minimumLength = minFrames;
        int64_t const boundedLength =
            roundedLength > minimumLength ? roundedLength : minimumLength;
        featureLengths[batch] = static_cast<int32_t>(
            boundedLength < numFrames ? boundedLength : numFrames);
    }

    extern __shared__ float frameSamples[];
    float sum = 0.0F;
    for (int32_t sample = thread; sample < frameLength; sample += blockDim.x)
    {
        int64_t const source =
            static_cast<int64_t>(frame) * frameShift + sample;
        int64_t const index =
            source < leftPadding ? leftPadding - 1 - source : source - leftPadding;
        float const value = audio[static_cast<int64_t>(batch) * audioSamples + index];
        frameSamples[sample] = value;
        sum += value;
    }
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    __shared__ float sums[kWarpsPerBlock];
    int32_t const lane = thread % kWarpSize;
    int32_t const warp = thread / kWarpSize;
    if (lane == 0)
    {
        sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        // The first warp reduces the eight per-warp totals. Every lane remains
        // active so the full-warp shuffle mask is valid.
        sum = lane < kWarpsPerBlock ? sums[lane] : 0.0F;
        for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
        }
        if (lane == 0)
        {
            sums[0] = sum;
        }
    }
    __syncthreads();
    float const mean = sums[0] / frameLength;
    int32_t const transformStride = fftLength + 2;
    for (int32_t sample = thread; sample < transformStride; sample += blockDim.x)
    {
        float value = 0.0F;
        if (sample < frameLength)
        {
            float const current = frameSamples[sample] - mean;
            if (sample == 0)
            {
                value = (1.0F - preemph) * current;
            }
            else
            {
                float const previous = frameSamples[sample - 1] - mean;
                value = current - preemph * previous;
            }
            value *= window[sample];
        }
        frames[static_cast<int64_t>(flattenedFrame) * transformStride + sample] = value;
    }
}

__global__ void powerSpectrumInPlace(
    float2 const* spectrum, float* power, int64_t rows, int32_t frequencies)
{
    // A complete complex row is staged before its compact power values overwrite
    // the same physical R2C row. The barriers prevent one thread from destroying
    // a complex component that another thread has not read yet.
    extern __shared__ float2 stagedSpectrum[];
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const transformStride = 2 * frequencies;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int64_t const spectrumOffset = row * frequencies;
        for (int32_t frequency = thread; frequency < frequencies; frequency += blockDim.x)
        {
            stagedSpectrum[frequency] = spectrum[spectrumOffset + frequency];
        }
        __syncthreads();
        int64_t const powerOffset = row * transformStride;
        for (int32_t frequency = thread; frequency < frequencies; frequency += blockDim.x)
        {
            float2 const value = stagedSpectrum[frequency];
            power[powerOffset + frequency] = value.x * value.x + value.y * value.y;
        }
        __syncthreads();
    }
}

__global__ void finalizeFeatures(float* features, int32_t const* featureLengths,
    int64_t elements, int32_t numFrames, int32_t numFeatures, float zeroLog)
{
    // Apply Kaldi's energy floor before log and overwrite model-padding frames
    // with the log value produced by an all-zero waveform.
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    while (index < elements)
    {
        int32_t const batch = static_cast<int32_t>(
            index / (static_cast<int64_t>(numFrames) * numFeatures));
        int32_t const frame = static_cast<int32_t>((index / numFeatures) % numFrames);
        int32_t const length = featureLengths[batch];
        features[index] = frame < length ? logf(fmaxf(features[index], 0x1p-23F)) : zeroLog;
        if (elements - index <= stride)
        {
            break;
        }
        index += stride;
    }
}

struct FeatureParameters
{
    int32_t frameLength{};
    int32_t frameShift{};
    int32_t leftPadding{};
    int32_t minFrames{};
    float preemph{};
    float zeroLog{};
};

bool areValidParameters(FeatureParameters const& parameters) noexcept
{
    return parameters.frameLength >= 2 && parameters.frameLength <= kMaxFrameLength
        && parameters.frameShift >= 1
        && parameters.frameShift <= parameters.frameLength && parameters.leftPadding >= 0
        && parameters.leftPadding < parameters.frameLength && parameters.minFrames >= 1
        && std::isfinite(parameters.preemph) && parameters.preemph >= 0.0F
        && parameters.preemph < 1.0F && std::isfinite(parameters.zeroLog);
}

bool haveValidShapes(Dims const& audio, Dims const& audioLengths,
    Dims const& window, Dims const& melFilterbank, Dims const& features,
    Dims const& featureLengths, FeatureParameters const& parameters) noexcept
{
    // Kernels and library calls use these descriptors for pointer arithmetic.
    // Validate the complete tensor relationship at every optimization-profile
    // point and again at runtime, including enough audio for the reflected
    // left context.
    if (audio.nbDims != 2 || audioLengths.nbDims != 1 || window.nbDims != 1
        || melFilterbank.nbDims != 2 || features.nbDims != 3
        || featureLengths.nbDims != 1
        || !hasAddressableBytes(audio, sizeof(float))
        || !hasAddressableBytes(audioLengths, sizeof(int64_t))
        || !hasAddressableBytes(window, sizeof(float))
        || !hasAddressableBytes(melFilterbank, sizeof(float))
        || !hasAddressableBytes(features, sizeof(float))
        || !hasAddressableBytes(featureLengths, sizeof(int32_t))
        || !areValidParameters(parameters)
        || audio.d[1] < parameters.leftPadding
        || audioLengths.d[0] != audio.d[0]
        || window.d[0] != parameters.frameLength || melFilterbank.d[0] < 2
        || melFilterbank.d[1] < 1 || features.d[0] != audio.d[0]
        || features.d[1] < parameters.minFrames
        || features.d[2] != melFilterbank.d[1]
        || featureLengths.d[0] != audio.d[0])
    {
        return false;
    }

    int64_t const frameNumerator = static_cast<int64_t>(audio.d[1])
        + parameters.leftPadding - parameters.frameLength;
    int64_t const expectedFrames =
        frameNumerator / parameters.frameShift + 1;
    int64_t const fftLength =
        2LL * (static_cast<int64_t>(melFilterbank.d[0]) - 1);
    int64_t const rows =
        static_cast<int64_t>(audio.d[0]) * features.d[1];
    WorkspaceLayout layout{};
    return frameNumerator >= 0 && expectedFrames == features.d[1]
        && fftLength >= parameters.frameLength
        && fftLength <= std::numeric_limits<int32_t>::max() - 2
        && melFilterbank.d[0] <= kMaxFrequencies && rows >= 1
        && rows <= std::numeric_limits<int32_t>::max()
        && makeWorkspaceLayout(rows, static_cast<int32_t>(fftLength),
            static_cast<int32_t>(melFilterbank.d[0]), layout);
}

// Convert normalized audio to Kaldi-compatible log-mel features. The plugin
// keeps framing and windowing, batched cuFFT, power extraction, mel projection,
// and padding finalization on one CUDA stream. Features remain FP32 throughout,
// and the second output contains valid frame counts for every waveform.
class ZipformerFeaturePlugin final : public IPluginV3,
                                     public IPluginV3OneCore,
                                     public IPluginV3OneBuild,
                                     public IPluginV3OneRuntime
{
public:
    explicit ZipformerFeaturePlugin(
        FeatureParameters parameters, int32_t tactic = kStrictComputeTactic) noexcept
        : mParameters(parameters), mTactic(tactic)
    {
        int const timingCacheLength = std::snprintf(mTimingCacheId.data(), mTimingCacheId.size(),
            "layout=inplace;frame_length=%d;frame_shift=%d;left_padding=%d;"
            "min_frames=%d;preemph=%a;zero_log=%a",
            parameters.frameLength, parameters.frameShift, parameters.leftPadding,
            parameters.minFrames, static_cast<double>(parameters.preemph),
            static_cast<double>(parameters.zeroLog));
        bool const validTimingCacheId = timingCacheLength > 0
            && static_cast<size_t>(timingCacheLength) < mTimingCacheId.size();
        mInitialized = areValidParameters(parameters) && validTimingCacheId
            && cublasCreate(&mCublas) == CUBLAS_STATUS_SUCCESS;
        initializeFields();
    }

    ~ZipformerFeaturePlugin() override
    {
        if (mPlan != 0)
        {
            cufftDestroy(mPlan);
        }
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
        auto* plugin =
            new (std::nothrow) ZipformerFeaturePlugin(mParameters, mTactic);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
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

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || inputs[0].desc.dims.nbDims != 2 || inputs[1].desc.dims.nbDims != 1
            || inputs[2].desc.dims.nbDims != 1 || inputs[3].desc.dims.nbDims != 2
            || outputs[0].desc.dims.nbDims != 3 || outputs[1].desc.dims.nbDims != 1
            || inputs[0].desc.type != DataType::kFLOAT
            || inputs[1].desc.type != DataType::kINT64
            || inputs[2].desc.type != DataType::kFLOAT
            || inputs[3].desc.type != DataType::kFLOAT
            || outputs[0].desc.type != DataType::kFLOAT
            || outputs[1].desc.type != DataType::kINT32
            || inputs[0].desc.format != TensorFormat::kLINEAR
            || inputs[1].desc.format != TensorFormat::kLINEAR
            || inputs[2].desc.format != TensorFormat::kLINEAR
            || inputs[3].desc.format != TensorFormat::kLINEAR
            || outputs[0].desc.format != TensorFormat::kLINEAR
            || outputs[1].desc.format != TensorFormat::kLINEAR || !mInitialized
            || !haveValidShapes(inputs[0].min, inputs[1].min, inputs[2].min,
                inputs[3].min, outputs[0].min, outputs[1].min, mParameters)
            || !haveValidShapes(inputs[0].opt, inputs[1].opt, inputs[2].opt,
                inputs[3].opt, outputs[0].opt, outputs[1].opt, mParameters)
            || !haveValidShapes(inputs[0].max, inputs[1].max, inputs[2].max,
                inputs[3].max, outputs[0].max, outputs[1].max, mParameters))
        {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbInputs != kInputCount || nbOutputs != kOutputCount
            || inputTypes[0] != DataType::kFLOAT || inputTypes[1] != DataType::kINT64
            || inputTypes[2] != DataType::kFLOAT || inputTypes[3] != DataType::kFLOAT)
        {
            return 1;
        }
        outputTypes[0] = DataType::kFLOAT;
        outputTypes[1] = DataType::kINT32;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs,
        int32_t nbOutputs, IExprBuilder& builder) noexcept override
    {
        static_cast<void>(shapeInputs);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || nbShapeInputs != 0 || inputs[0].nbDims != 2 || inputs[1].nbDims != 1
            || inputs[2].nbDims != 1 || inputs[3].nbDims != 2
            || inputs[0].d[0] == nullptr || inputs[0].d[1] == nullptr
            || inputs[1].d[0] == nullptr || inputs[2].d[0] == nullptr
            || inputs[3].d[0] == nullptr || inputs[3].d[1] == nullptr
            || !mInitialized)
        {
            return 1;
        }
        auto* offset = builder.constant(mParameters.leftPadding - mParameters.frameLength);
        auto* frameShift = builder.constant(mParameters.frameShift);
        auto* one = builder.constant(1);
        if (offset == nullptr || frameShift == nullptr || one == nullptr)
        {
            return 1;
        }
        auto* numerator =
            builder.operation(DimensionOperation::kSUM, *inputs[0].d[1], *offset);
        if (numerator == nullptr)
        {
            return 1;
        }
        auto* frames =
            builder.operation(DimensionOperation::kFLOOR_DIV, *numerator, *frameShift);
        if (frames == nullptr)
        {
            return 1;
        }
        frames = builder.operation(DimensionOperation::kSUM, *frames, *one);
        if (frames == nullptr)
        {
            return 1;
        }
        outputs[0].nbDims = 3;
        // Frames = floor((samples + leftPadding - frameLength) / frameShift) + 1.
        // The mel-filterbank column count is the feature dimension.
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = frames;
        outputs[0].d[2] = inputs[3].d[1];
        outputs[1] = inputs[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || pos < 0
            || pos >= kInputCount + kOutputCount)
        {
            return false;
        }
        auto const type = pos == 1 ? DataType::kINT64
            : (pos == 5 ? DataType::kINT32 : DataType::kFLOAT);
        return inOut[pos].desc.format == TensorFormat::kLINEAR && inOut[pos].desc.type == type;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        // TensorRT allocates one workspace for the maximum profile shape. The
        // runtime layout for smaller shapes is always a prefix of this buffer.
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || !haveValidShapes(inputs[0].max, inputs[1].max, inputs[2].max,
                inputs[3].max, outputs[0].max, outputs[1].max, mParameters))
        {
            return 0;
        }
        auto const& audio = inputs[0].max;
        auto const& features = outputs[0].max;
        int64_t const rows = static_cast<int64_t>(audio.d[0]) * features.d[1];
        int64_t const frequencies64 = inputs[3].max.d[0];
        int64_t const fftLength64 = 2LL * (frequencies64 - 1);
        WorkspaceLayout layout{};
        return makeWorkspaceLayout(rows, static_cast<int32_t>(fftLength64),
                   static_cast<int32_t>(frequencies64), layout)
            ? layout.totalBytes : 0;
    }

    int32_t getNbTactics() noexcept override
    {
        return deviceSupportsAmpereCompute()
            ? static_cast<int32_t>(kMelProjectionTactics.size()) : 1;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        if (tactics == nullptr
            || nbTactics != getNbTactics())
        {
            return 1;
        }
        std::copy_n(kMelProjectionTactics.begin(), nbTactics, tactics);
        return 0;
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic == 0)
        {
            tactic = kStrictComputeTactic;
        }
        if (!isMelProjectionTactic(tactic))
        {
            return 1;
        }
        mTactic = tactic;
        return 0;
    }

    char const* getTimingCacheID() noexcept override
    {
        // TensorRT combines this parameter identity with tensor shapes and
        // formats, allowing equivalent frontend instances to reuse tactic timing.
        return mTimingCacheId.data();
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || inputs[0].type != DataType::kFLOAT
            || inputs[1].type != DataType::kINT64
            || inputs[2].type != DataType::kFLOAT
            || inputs[3].type != DataType::kFLOAT
            || outputs[0].type != DataType::kFLOAT
            || outputs[1].type != DataType::kINT32
            || inputs[0].format != TensorFormat::kLINEAR
            || inputs[1].format != TensorFormat::kLINEAR
            || inputs[2].format != TensorFormat::kLINEAR
            || inputs[3].format != TensorFormat::kLINEAR
            || outputs[0].format != TensorFormat::kLINEAR
            || outputs[1].format != TensorFormat::kLINEAR
            || !isMelProjectionTactic(mTactic) || !mInitialized
            || !haveValidShapes(inputs[0].dims, inputs[1].dims, inputs[2].dims,
                inputs[3].dims, outputs[0].dims, outputs[1].dims, mParameters))
        {
            return 1;
        }

        int64_t const frequencies = inputs[3].dims.d[0];
        int64_t const fftLength64 = 2LL * (frequencies - 1);
        int64_t const rows64 =
            static_cast<int64_t>(inputs[0].dims.d[0]) * outputs[0].dims.d[1];
        int32_t const rows = static_cast<int32_t>(rows64);
        int32_t const fftLength = static_cast<int32_t>(fftLength64);
        if (mPlan != 0 && rows == mPlanRows && fftLength == mPlanLength)
        {
            return 0;
        }
        if (mPlan != 0)
        {
            cufftDestroy(mPlan);
        }
        mPlan = 0;
        mPlanRows = 0;
        mPlanLength = 0;
        mCufftStream = nullptr;
        mCublasWorkspace = nullptr;
        // A cuFFT plan fixes both transform length and batch count. Construct a
        // replacement before publishing its dimensions; a failed plan therefore
        // cannot be mistaken for a valid cached plan by enqueue().
        int32_t const planFrequencies = fftLength / 2 + 1;
        int32_t const transformStride = fftLength + 2;
        int32_t dimensions[]{fftLength};
        int32_t inputEmbed[]{transformStride};
        int32_t outputEmbed[]{planFrequencies};
        cufftResult const result = cufftPlanMany(&mPlan, 1, dimensions, inputEmbed, 1,
            transformStride, outputEmbed, 1, planFrequencies, CUFFT_R2C, rows);
        if (result != CUFFT_SUCCESS)
        {
            if (mPlan != 0)
            {
                cufftDestroy(mPlan);
                mPlan = 0;
            }
            return 1;
        }
        mPlanRows = rows;
        mPlanLength = fftLength;
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr
            || inputs[2] == nullptr || inputs[3] == nullptr || outputs[0] == nullptr
            || outputs[1] == nullptr || workspace == nullptr || mPlan == 0
            || !mInitialized || inputDesc[0].type != DataType::kFLOAT
            || inputDesc[1].type != DataType::kINT64
            || inputDesc[2].type != DataType::kFLOAT
            || inputDesc[3].type != DataType::kFLOAT
            || outputDesc[0].type != DataType::kFLOAT
            || outputDesc[1].type != DataType::kINT32
            || inputDesc[0].format != TensorFormat::kLINEAR
            || inputDesc[1].format != TensorFormat::kLINEAR
            || inputDesc[2].format != TensorFormat::kLINEAR
            || inputDesc[3].format != TensorFormat::kLINEAR
            || outputDesc[0].format != TensorFormat::kLINEAR
            || outputDesc[1].format != TensorFormat::kLINEAR
            || !isMelProjectionTactic(mTactic)
            || !haveValidShapes(inputDesc[0].dims, inputDesc[1].dims,
                inputDesc[2].dims, inputDesc[3].dims, outputDesc[0].dims,
                outputDesc[1].dims, mParameters))
        {
            return 1;
        }

        int32_t const batch = inputDesc[0].dims.d[0];
        int32_t const audioSamples = inputDesc[0].dims.d[1];
        int32_t const numFrames = outputDesc[0].dims.d[1];
        int32_t const numFeatures = outputDesc[0].dims.d[2];
        int32_t const frequencies = inputDesc[3].dims.d[0];
        int32_t const fftLength = 2 * (frequencies - 1);
        int32_t const transformStride = fftLength + 2;
        int64_t const rows = static_cast<int64_t>(batch) * numFrames;
        WorkspaceLayout layout{};
        if (batch < 1 || audioSamples < 1 || numFrames < mParameters.minFrames
            || numFeatures < 1 || frequencies < 2 || rows != mPlanRows
            || fftLength != mPlanLength
            || !makeWorkspaceLayout(rows, fftLength, frequencies, layout))
        {
            return 1;
        }

        // Preserve errors from earlier asynchronous work instead of silently
        // attributing success to this invocation.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        auto* workspaceBytes = static_cast<unsigned char*>(workspace);
        auto* frames = reinterpret_cast<float*>(workspaceBytes);
        // Frames, in-place complex spectra, and padded power rows intentionally
        // share this transform buffer across their ordered stream stages.
        auto* power = frames;
        auto* cublasWorkspace = workspaceBytes + layout.cublasOffset;

        // A rebuilt cuFFT plan starts on stream 0, while the independent cuBLAS
        // handle retains its previous stream. Track both bindings separately so
        // a dynamic shape change followed by stream 0 cannot leave cuBLAS racing
        // the plan and custom kernels on an older nondefault stream.
        if (stream != mCufftStream)
        {
            if (cufftSetStream(mPlan, stream) != CUFFT_SUCCESS)
            {
                return 1;
            }
            mCufftStream = stream;
        }
        if (stream != mCublasStream)
        {
            if (cublasSetStream(mCublas, stream) != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mCublasStream = stream;
            // cublasSetStream resets the handle workspace even when TensorRT
            // reuses the same workspace address across executions.
            mCublasWorkspace = nullptr;
        }

        prepareFrames<<<static_cast<uint32_t>(rows), kThreadsPerBlock,
            static_cast<size_t>(mParameters.frameLength) * sizeof(float), stream>>>(
            static_cast<float const*>(inputs[0]), static_cast<int64_t const*>(inputs[1]),
            static_cast<float const*>(inputs[2]), frames, static_cast<int32_t*>(outputs[1]),
            audioSamples, numFrames, mParameters.frameLength, mParameters.frameShift,
            mParameters.leftPadding, fftLength, mParameters.minFrames, mParameters.preemph);
        if (cudaGetLastError() != cudaSuccess
            || cufftExecR2C(mPlan, frames, reinterpret_cast<cufftComplex*>(frames))
                != CUFFT_SUCCESS)
        {
            return 1;
        }
        int32_t const powerBlocks = static_cast<int32_t>(std::min<int64_t>(
            kMaxBlocks, rows));
        powerSpectrumInPlace<<<powerBlocks, kThreadsPerBlock,
            static_cast<size_t>(frequencies) * sizeof(cufftComplex), stream>>>(
            reinterpret_cast<cufftComplex const*>(frames), power, rows, frequencies);
        if (cudaGetLastError() != cudaSuccess)
        {
            return 1;
        }
        // TensorRT normally reuses one context workspace across executions.
        // Avoid repeating the host-side cuBLAS update unless either its base
        // allocation or this dynamic shape's aligned FFT prefix changed.
        if (cublasWorkspace != mCublasWorkspace)
        {
            if (cublasSetWorkspace(mCublas, cublasWorkspace, kCublasWorkspaceBytes)
                != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mCublasWorkspace = cublasWorkspace;
        }
        float const alpha = 1.0F;
        float const beta = 0.0F;
        // Logical row-major [rows, frequencies] power uses the padded in-place
        // transform stride. Together with row-major [frequencies, features] mel
        // storage it forms the desired product when viewed as transposed
        // column-major matrices, so cuBLAS writes [rows, features] without copies.
        cublasStatus_t const gemm = cublasGemmEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N,
            numFeatures, static_cast<int32_t>(rows), frequencies, &alpha, inputs[3],
            CUDA_R_32F, numFeatures, power, CUDA_R_32F, transformStride, &beta, outputs[0],
            CUDA_R_32F, numFeatures, getCublasComputeType(mTactic), CUBLAS_GEMM_DEFAULT);
        if (gemm != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }
        int64_t const featureElements = rows * numFeatures;
        int32_t const finalizeBlocks = static_cast<int32_t>(std::min<int64_t>(
            kMaxBlocks, featureElements / kThreadsPerBlock
                + (featureElements % kThreadsPerBlock != 0)));
        finalizeFeatures<<<finalizeBlocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<float*>(outputs[0]), static_cast<int32_t const*>(outputs[1]),
            featureElements, numFrames, numFeatures, mParameters.zeroLog);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        // Every TensorRT execution context receives independent cuFFT/cuBLAS
        // handles and stream state, so concurrent contexts cannot race.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    friend class ZipformerFeaturePluginCreator;

    void initializeFields() noexcept
    {
        mSerializedFields = {{
            {kFrameLengthField, &mParameters.frameLength, PluginFieldType::kINT32, 1},
            {kFrameShiftField, &mParameters.frameShift, PluginFieldType::kINT32, 1},
            {kLeftPaddingField, &mParameters.leftPadding, PluginFieldType::kINT32, 1},
            {kMinFramesField, &mParameters.minFrames, PluginFieldType::kINT32, 1},
            {kPreemphField, &mParameters.preemph, PluginFieldType::kFLOAT32, 1},
            {kZeroLogField, &mParameters.zeroLog, PluginFieldType::kFLOAT32, 1},
        }};
        mFields = {static_cast<int32_t>(mSerializedFields.size()), mSerializedFields.data()};
    }

    FeatureParameters mParameters{};
    cublasHandle_t mCublas{nullptr};
    cufftHandle mPlan{};
    cudaStream_t mCublasStream{nullptr};
    cudaStream_t mCufftStream{nullptr};
    void* mCublasWorkspace{nullptr};
    int32_t mPlanRows{};
    int32_t mPlanLength{};
    int32_t mTactic{kStrictComputeTactic};
    bool mInitialized{false};
    std::array<char, kTimingCacheIdSize> mTimingCacheId{};
    std::array<PluginField, 6> mSerializedFields{};
    PluginFieldCollection mFields{};
};

class ZipformerFeaturePluginCreator final : public IPluginCreatorV3One
{
public:
    ZipformerFeaturePluginCreator() noexcept
    {
        mAttributes = {{
            {kFrameLengthField, nullptr, PluginFieldType::kINT32, 1},
            {kFrameShiftField, nullptr, PluginFieldType::kINT32, 1},
            {kLeftPaddingField, nullptr, PluginFieldType::kINT32, 1},
            {kMinFramesField, nullptr, PluginFieldType::kINT32, 1},
            {kPreemphField, nullptr, PluginFieldType::kFLOAT32, 1},
            {kZeroLogField, nullptr, PluginFieldType::kFLOAT32, 1},
        }};
        mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
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

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fields,
        TensorRTPhase phase) noexcept override
    {
        static_cast<void>(name);
        static_cast<void>(phase);
        if (fields == nullptr || fields->nbFields < 0
            || (fields->nbFields > 0 && fields->fields == nullptr))
        {
            return nullptr;
        }

        FeatureParameters parameters{};
        uint32_t foundFields = 0;
        constexpr uint32_t allFields = (1U << 6) - 1;
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name == nullptr)
            {
                continue;
            }
            std::string_view const fieldName(field.name);
            uint32_t fieldIndex{};
            PluginFieldType expectedType{};
            void* destination{};
            if (fieldName == kFrameLengthField)
            {
                fieldIndex = 0;
                expectedType = PluginFieldType::kINT32;
                destination = &parameters.frameLength;
            }
            else if (fieldName == kFrameShiftField)
            {
                fieldIndex = 1;
                expectedType = PluginFieldType::kINT32;
                destination = &parameters.frameShift;
            }
            else if (fieldName == kLeftPaddingField)
            {
                fieldIndex = 2;
                expectedType = PluginFieldType::kINT32;
                destination = &parameters.leftPadding;
            }
            else if (fieldName == kMinFramesField)
            {
                fieldIndex = 3;
                expectedType = PluginFieldType::kINT32;
                destination = &parameters.minFrames;
            }
            else if (fieldName == kPreemphField)
            {
                fieldIndex = 4;
                expectedType = PluginFieldType::kFLOAT32;
                destination = &parameters.preemph;
            }
            else if (fieldName == kZeroLogField)
            {
                fieldIndex = 5;
                expectedType = PluginFieldType::kFLOAT32;
                destination = &parameters.zeroLog;
            }
            else
            {
                // TensorRT may append implementation metadata. Ignore unknown
                // fields while requiring each frontend parameter exactly once.
                continue;
            }

            uint32_t const fieldBit = 1U << fieldIndex;
            if ((foundFields & fieldBit) != 0 || field.type != expectedType
                || field.length != 1 || field.data == nullptr)
            {
                return nullptr;
            }
            if (expectedType == PluginFieldType::kINT32)
            {
                *static_cast<int32_t*>(destination) =
                    *static_cast<int32_t const*>(field.data);
            }
            else
            {
                *static_cast<float*>(destination) =
                    *static_cast<float const*>(field.data);
            }
            foundFields |= fieldBit;
        }
        if (foundFields != allFields || !areValidParameters(parameters))
        {
            return nullptr;
        }

        auto* plugin = new (std::nothrow) ZipformerFeaturePlugin(parameters);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

private:
    std::array<PluginField, 6> mAttributes{};
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrZipformerFeaturePlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Runtime and builder use separate registries. Treat an existing creator as
    // success so repeated package imports remain harmless.
    static ZipformerFeaturePluginCreator runtimeCreator;
    static ZipformerFeaturePluginCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry,
                                ZipformerFeaturePluginCreator& creator) noexcept {
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
