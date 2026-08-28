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

constexpr char const* kPluginName = "parakeet_feature_extractor";
constexpr char const* kPluginVersion = "1";
constexpr int32_t kInputCount = 4;
constexpr int32_t kOutputCount = 2;
constexpr char const* kFrameShiftField = "frame_shift";
constexpr char const* kPreemphField = "preemph";
constexpr char const* kLogEpsField = "log_eps";
constexpr char const* kEpsField = "eps";
constexpr size_t kCublasWorkspaceBytes = 16U << 20;
constexpr size_t kCublasWorkspaceAlignment = 256;
constexpr size_t kMaximumWorkspaceBytes =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kParallelNormalizationThreads = 64;
constexpr int32_t kCoalescedNormalizationThreads = 128;
constexpr int32_t kMaxBlocks = 65535;
constexpr size_t kTimingCacheIdSize = 160;
constexpr size_t kPortableSharedMemoryBytes = 48U << 10;
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
static_assert(
    (kParallelNormalizationThreads & (kParallelNormalizationThreads - 1)) == 0);

bool isMelProjectionTactic(int32_t tactic) noexcept
{
    return std::find(kMelProjectionTactics.begin(), kMelProjectionTactics.end(),
               tactic)
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
    int64_t rows, int32_t fftLength, WorkspaceLayout& layout) noexcept
{
    if (rows < 1 || fftLength < 2 || fftLength % 2 != 0
        || fftLength > std::numeric_limits<int32_t>::max() - 2)
    {
        return false;
    }

    // An in-place R2C transform needs two padding floats after every even-sized
    // real row. The same allocation subsequently stores strided power rows,
    // keeping the maximum production workspace below TensorRT's 2 GiB boundary.
    int32_t const transformStride = fftLength + 2;
    size_t transformElements{};
    size_t transformBytes{};
    if (!(checkedMultiply(static_cast<size_t>(rows),
              static_cast<size_t>(transformStride), transformElements)
            && checkedMultiply(transformElements, sizeof(float), transformBytes)
            && checkedAlignUp(transformBytes, kCublasWorkspaceAlignment,
                layout.cublasOffset)
            && checkedAdd(layout.cublasOffset, kCublasWorkspaceBytes,
                layout.totalBytes)))
    {
        return false;
    }
    // TensorRT's surrounding ForeignNode path uses signed 32-bit workspace byte
    // offsets on supported releases. Reject oversized profiles during build
    // instead of allowing an internal offset to wrap during tactic execution.
    return layout.totalBytes <= kMaximumWorkspaceBytes;
}

struct FeatureParameters
{
    int32_t frameShift{};
    float preemph{};
    float logEps{};
    float eps{};
};

bool areValidParameters(FeatureParameters const& parameters) noexcept
{
    return parameters.frameShift >= 1 && std::isfinite(parameters.preemph)
        && parameters.preemph >= 0.0F && parameters.preemph < 1.0F
        && std::isfinite(parameters.logEps) && parameters.logEps > 0.0F
        && std::isfinite(parameters.eps) && parameters.eps > 0.0F;
}

bool haveValidShapes(Dims const& audio, Dims const& audioLengths,
    Dims const& window, Dims const& melFilterbank, Dims const& features,
    Dims const& featureLengths, FeatureParameters const& parameters) noexcept
{
    // The plugin owns every pointer calculation made by its kernels, cuFFT,
    // and cuBLAS. Validate the complete tensor relationship at each profile
    // point and again for every concrete runtime shape.
    if (audio.nbDims != 2 || audioLengths.nbDims != 1 || window.nbDims != 1
        || melFilterbank.nbDims != 2 || features.nbDims != 3
        || featureLengths.nbDims != 1
        || !hasAddressableBytes(audio, sizeof(float))
        || !hasAddressableBytes(audioLengths, sizeof(int64_t))
        || !hasAddressableBytes(window, sizeof(float))
        || !hasAddressableBytes(melFilterbank, sizeof(float))
        || !hasAddressableBytes(features, sizeof(float))
        || !hasAddressableBytes(featureLengths, sizeof(int32_t))
        || !areValidParameters(parameters) || audioLengths.d[0] != audio.d[0]
        || featureLengths.d[0] != audio.d[0] || features.d[0] != audio.d[0]
        || window.d[0] < 2 || window.d[0] % 2 != 0
        || window.d[0] > std::numeric_limits<int32_t>::max() - 2
        || melFilterbank.d[0] != window.d[0] / 2 + 1
        || melFilterbank.d[0] > kMaxFrequencies
        || features.d[2] != melFilterbank.d[1])
    {
        return false;
    }

    int64_t const expectedFrames =
        audio.d[1] / parameters.frameShift + 1;
    int64_t const rows = static_cast<int64_t>(audio.d[0]) * expectedFrames;
    int64_t const normalizationBlocks =
        static_cast<int64_t>(audio.d[0]) * features.d[2];
    WorkspaceLayout layout{};
    return expectedFrames == features.d[1] && rows >= 1
        && rows <= std::numeric_limits<int32_t>::max()
        && normalizationBlocks >= 1
        && normalizationBlocks <= std::numeric_limits<int32_t>::max()
        && makeWorkspaceLayout(rows, static_cast<int32_t>(window.d[0]), layout);
}

__global__ void prepareFrames(float const* audio, int64_t const* audioLengths,
    float const* window, float* frames, int32_t audioSamples,
    int32_t numFrames, int32_t fftLength, int32_t frameShift, float preemph)
{
    // One block owns one centered STFT frame. Invalid waveform padding and the
    // centered left/right context are explicitly zeroed before the in-place FFT.
    int32_t const flattenedFrame = static_cast<int32_t>(blockIdx.x);
    int32_t const batch = flattenedFrame / numFrames;
    int32_t const frame = flattenedFrame - batch * numFrames;
    int64_t validSamples = audioLengths[batch];
    validSamples = validSamples < 0
        ? 0
        : (validSamples > audioSamples ? audioSamples : validSamples);
    int64_t const frameStart =
        static_cast<int64_t>(frame) * frameShift - fftLength / 2;
    int32_t const transformStride = fftLength + 2;

    for (int32_t sample = static_cast<int32_t>(threadIdx.x);
         sample < transformStride; sample += blockDim.x)
    {
        float value = 0.0F;
        if (sample < fftLength)
        {
            int64_t const source = frameStart + sample;
            if (source >= 0 && source < validSamples)
            {
                int64_t const offset =
                    static_cast<int64_t>(batch) * audioSamples + source;
                value = audio[offset];
                if (source > 0)
                {
                    value -= preemph * audio[offset - 1];
                }
                value *= window[sample];
            }
        }
        frames[static_cast<int64_t>(flattenedFrame) * transformStride + sample]
            = value;
    }
}

__global__ void powerSpectrumInPlace(
    float2 const* spectrum, float* power, int64_t rows, int32_t frequencies)
{
    // Generic FFT sizes stage a complete complex row before compact power values
    // overwrite its physical in-place R2C row.
    extern __shared__ float2 stagedSpectrum[];
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const transformStride = 2 * frequencies;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int64_t const spectrumOffset = row * frequencies;
        for (int32_t frequency = thread; frequency < frequencies;
             frequency += blockDim.x)
        {
            stagedSpectrum[frequency] = spectrum[spectrumOffset + frequency];
        }
        __syncthreads();
        int64_t const powerOffset = row * transformStride;
        for (int32_t frequency = thread; frequency < frequencies;
             frequency += blockDim.x)
        {
            float2 const value = stagedSpectrum[frequency];
            power[powerOffset + frequency] =
                value.x * value.x + value.y * value.y;
        }
        __syncthreads();
    }
}

__global__ void normalizeFeaturesParallel(float* features,
    int64_t const* audioLengths, int32_t* featureLengths,
    int32_t audioSamples, int32_t numFrames, int32_t numFeatures,
    int32_t frameShift, float logEps, float eps)
{
    // Small batches need feature-level blocks to expose enough parallel work.
    // Welford states are merged across a fixed power-of-two thread block.
    int32_t const batchFeature = static_cast<int32_t>(blockIdx.x);
    int32_t const batch = batchFeature / numFeatures;
    int32_t const feature = batchFeature - batch * numFeatures;
    int64_t validSamples = audioLengths[batch];
    validSamples = validSamples < 0
        ? 0 : (validSamples > audioSamples ? audioSamples : validSamples);
    int32_t const length = static_cast<int32_t>(validSamples / frameShift);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);

    if (length < 2)
    {
        for (int32_t frame = thread; frame < numFrames; frame += blockDim.x)
        {
            int64_t const index =
                (static_cast<int64_t>(batch) * numFrames + frame) * numFeatures
                + feature;
            features[index] = 0.0F;
        }
        if (feature == 0 && thread == 0)
        {
            featureLengths[batch] = length;
        }
        return;
    }

    int32_t localCount = 0;
    float localMean = 0.0F;
    float localM2 = 0.0F;
    for (int32_t frame = thread; frame < length; frame += blockDim.x)
    {
        int64_t const index =
            (static_cast<int64_t>(batch) * numFrames + frame) * numFeatures
            + feature;
        float const value = logf(features[index] + logEps);
        features[index] = value;
        ++localCount;
        float const delta = value - localMean;
        localMean += delta / localCount;
        localM2 += delta * (value - localMean);
    }

    __shared__ int32_t counts[kParallelNormalizationThreads];
    __shared__ float means[kParallelNormalizationThreads];
    __shared__ float m2s[kParallelNormalizationThreads];
    counts[thread] = localCount;
    means[thread] = localMean;
    m2s[thread] = localM2;
    __syncthreads();
    for (int32_t width = blockDim.x / 2; width > 0; width /= 2)
    {
        if (thread < width)
        {
            int32_t const otherCount = counts[thread + width];
            if (otherCount > 0)
            {
                int32_t const count = counts[thread];
                if (count == 0)
                {
                    counts[thread] = otherCount;
                    means[thread] = means[thread + width];
                    m2s[thread] = m2s[thread + width];
                }
                else
                {
                    int32_t const combinedCount = count + otherCount;
                    float const delta = means[thread + width] - means[thread];
                    means[thread] += delta * otherCount / combinedCount;
                    m2s[thread] += m2s[thread + width]
                        + delta * delta * static_cast<float>(count)
                            * static_cast<float>(otherCount)
                            / static_cast<float>(combinedCount);
                    counts[thread] = combinedCount;
                }
            }
        }
        __syncthreads();
    }

    float const mean = means[0];
    float const variance = fmaxf(m2s[0], 0.0F) / (length - 1);
    float const standardDeviation = sqrtf(variance) + eps;
    for (int32_t frame = thread; frame < numFrames; frame += blockDim.x)
    {
        int64_t const index =
            (static_cast<int64_t>(batch) * numFrames + frame) * numFeatures
            + feature;
        features[index] = frame < length
            ? (features[index] - mean) / standardDeviation
            : 0.0F;
    }
    if (feature == 0 && thread == 0)
    {
        featureLengths[batch] = length;
    }
}

__global__ void normalizeFeaturesCoalesced(float* features,
    int64_t const* audioLengths, int32_t* featureLengths,
    int32_t audioSamples, int32_t numFrames, int32_t numFeatures,
    int32_t frameShift, float logEps, float eps)
{
    // One block owns one utterance and neighboring threads own neighboring mel
    // channels. Every frame therefore produces coalesced feature loads/stores,
    // while each thread keeps its numerically stable Welford state in registers.
    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int64_t validSamples = audioLengths[batch];
    validSamples = validSamples < 0
        ? 0
        : (validSamples > audioSamples ? audioSamples : validSamples);
    int32_t const length = static_cast<int32_t>(validSamples / frameShift);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    if (thread == 0)
    {
        featureLengths[batch] = length;
    }

    // Production validation guarantees at least two valid frames. Keep the
    // plugin memory-safe and deterministic if a caller supplies malformed
    // device-side lengths that TensorRT cannot inspect during shape validation.
    for (int32_t feature = thread; feature < numFeatures;
         feature += blockDim.x)
    {
        if (length < 2)
        {
            for (int32_t frame = 0; frame < numFrames; ++frame)
            {
                int64_t const index =
                    (static_cast<int64_t>(batch) * numFrames + frame)
                        * numFeatures
                    + feature;
                features[index] = 0.0F;
            }
            continue;
        }

        int32_t count = 0;
        float mean = 0.0F;
        float m2 = 0.0F;
        for (int32_t frame = 0; frame < length; ++frame)
        {
            int64_t const index =
                (static_cast<int64_t>(batch) * numFrames + frame) * numFeatures
                + feature;
            float const value = logf(features[index] + logEps);
            features[index] = value;
            ++count;
            float const delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
        }
        float const variance = fmaxf(m2, 0.0F) / (length - 1);
        float const standardDeviation = sqrtf(variance) + eps;
        for (int32_t frame = 0; frame < numFrames; ++frame)
        {
            int64_t const index =
                (static_cast<int64_t>(batch) * numFrames + frame) * numFeatures
                + feature;
            features[index] = frame < length
                ? (features[index] - mean) / standardDeviation
                : 0.0F;
        }
    }
}

// Convert padded mono audio to NeMo-compatible normalized log-mel features.
// Framing, an in-place batched cuFFT, power extraction, mel projection, and
// per-feature sample-standard-deviation normalization remain on one CUDA stream.
class ParakeetFeaturePlugin final : public IPluginV3,
                                    public IPluginV3OneCore,
                                    public IPluginV3OneBuild,
                                    public IPluginV3OneRuntime
{
public:
    explicit ParakeetFeaturePlugin(
        FeatureParameters parameters, int32_t tactic = kStrictComputeTactic) noexcept
        : mParameters(parameters), mTactic(tactic)
    {
        mInitialized = areValidParameters(parameters)
            && cublasCreate(&mCublas) == CUBLAS_STATUS_SUCCESS;
        if (mInitialized)
        {
            int32_t device{};
            int32_t multiprocessors{};
            if (cudaGetDevice(&device) == cudaSuccess
                && cudaDeviceGetAttribute(
                       &multiprocessors, cudaDevAttrMultiProcessorCount, device)
                    == cudaSuccess
                && multiprocessors > 0)
            {
                mCoalescedNormalizationMinBatch = multiprocessors;
            }
            else
            {
                // Device discovery only selects a performance crossover. Keep
                // the portable feature-parallel path and clear the optional
                // query error so it cannot leak into a later kernel check.
                cudaGetLastError();
            }
        }
        int const timingCacheLength = std::snprintf(mTimingCacheId.data(),
            mTimingCacheId.size(),
            "layout=inplace_fft;normalization=adaptive;frame_shift=%d;"
            "normalization_switch=%d;preemph=%a;log_eps=%a;eps=%a",
            parameters.frameShift, mCoalescedNormalizationMinBatch,
            static_cast<double>(parameters.preemph),
            static_cast<double>(parameters.logEps),
            static_cast<double>(parameters.eps));
        bool const validTimingCacheId = timingCacheLength > 0
            && static_cast<size_t>(timingCacheLength) < mTimingCacheId.size();
        mInitialized = mInitialized && validTimingCacheId;
        initializeFields();
    }

    ~ParakeetFeaturePlugin() override
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
        auto* plugin =
            new (std::nothrow) ParakeetFeaturePlugin(mParameters, mTactic);
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

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || inputs[0].desc.dims.nbDims != 2
            || inputs[1].desc.dims.nbDims != 1
            || inputs[2].desc.dims.nbDims != 1
            || inputs[3].desc.dims.nbDims != 2
            || outputs[0].desc.dims.nbDims != 3
            || outputs[1].desc.dims.nbDims != 1
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
            || outputs[1].desc.format != TensorFormat::kLINEAR
            || !mInitialized
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
            || inputTypes[0] != DataType::kFLOAT
            || inputTypes[1] != DataType::kINT64
            || inputTypes[2] != DataType::kFLOAT
            || inputTypes[3] != DataType::kFLOAT)
        {
            return 1;
        }
        outputTypes[0] = DataType::kFLOAT;
        outputTypes[1] = DataType::kINT32;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& builder) noexcept override
    {
        static_cast<void>(shapeInputs);
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || nbShapeInputs != 0
            || inputs[0].nbDims != 2
            || inputs[1].nbDims != 1 || inputs[2].nbDims != 1
            || inputs[3].nbDims != 2 || inputs[0].d[0] == nullptr
            || inputs[0].d[1] == nullptr || inputs[1].d[0] == nullptr
            || inputs[2].d[0] == nullptr || inputs[3].d[0] == nullptr
            || inputs[3].d[1] == nullptr || !mInitialized)
        {
            return 1;
        }
        auto* frameShift = builder.constant(mParameters.frameShift);
        auto* one = builder.constant(1);
        if (frameShift == nullptr || one == nullptr)
        {
            return 1;
        }
        auto* frames = builder.operation(
            DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *frameShift);
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
        // Centered padding produces floor(samples / frameShift) + 1 frames.
        // The mel-filterbank column count is the output feature dimension.
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = frames;
        outputs[0].d[2] = inputs[3].d[1];
        outputs[1] = inputs[1];
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
        auto const type = pos == 1 ? DataType::kINT64
            : (pos == 5 ? DataType::kINT32 : DataType::kFLOAT);
        return inOut[pos].desc.format == TensorFormat::kLINEAR
            && inOut[pos].desc.type == type;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        // TensorRT allocates one workspace for the maximum profile shape; all
        // smaller runtime layouts are prefixes of that allocation.
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount
            || !haveValidShapes(inputs[0].max, inputs[1].max, inputs[2].max,
                inputs[3].max, outputs[0].max, outputs[1].max, mParameters))
        {
            return 0;
        }
        int64_t const rows = static_cast<int64_t>(inputs[0].max.d[0])
            * outputs[0].max.d[1];
        WorkspaceLayout layout{};
        return makeWorkspaceLayout(
                   rows, static_cast<int32_t>(inputs[2].max.d[0]), layout)
            ? layout.totalBytes
            : 0;
    }

    int32_t getNbTactics() noexcept override
    {
        return static_cast<int32_t>(kMelProjectionTactics.size());
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        if (tactics == nullptr
            || nbTactics != static_cast<int32_t>(kMelProjectionTactics.size()))
        {
            return 1;
        }
        std::copy(kMelProjectionTactics.begin(), kMelProjectionTactics.end(), tactics);
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
        // TensorRT combines this identity with shapes and formats, allowing
        // equivalent frontend instances to share tactic measurements.
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
            || !haveValidShapes(inputs[0].dims, inputs[1].dims,
                inputs[2].dims, inputs[3].dims, outputs[0].dims,
                outputs[1].dims, mParameters))
        {
            return 1;
        }

        int32_t const rows = static_cast<int32_t>(
            static_cast<int64_t>(inputs[0].dims.d[0]) * outputs[0].dims.d[1]);
        int32_t const fftLength = static_cast<int32_t>(inputs[2].dims.d[0]);
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
        mStream = nullptr;
        mStreamInitialized = false;
        mCublasWorkspace = nullptr;

        // A plan fixes transform length, physical row stride, and batch count.
        // Publish its cached dimensions only after successful construction so a
        // failed plan can never be mistaken for a reusable one.
        int32_t const frequencies = fftLength / 2 + 1;
        int32_t const transformStride = fftLength + 2;
        int32_t dimensions[]{fftLength};
        int32_t inputEmbed[]{transformStride};
        int32_t outputEmbed[]{frequencies};
        cufftResult const result = cufftPlanMany(&mPlan, 1, dimensions,
            inputEmbed, 1, transformStride, outputEmbed, 1, frequencies,
            CUFFT_R2C, rows);
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

    int32_t enqueue(PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr
            || inputs[1] == nullptr || inputs[2] == nullptr
            || inputs[3] == nullptr || outputs[0] == nullptr
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

        int32_t const batch = static_cast<int32_t>(inputDesc[0].dims.d[0]);
        int32_t const audioSamples = static_cast<int32_t>(inputDesc[0].dims.d[1]);
        int32_t const fftLength = static_cast<int32_t>(inputDesc[2].dims.d[0]);
        int32_t const numFrames = static_cast<int32_t>(outputDesc[0].dims.d[1]);
        int32_t const numFeatures = static_cast<int32_t>(outputDesc[0].dims.d[2]);
        int32_t const frequencies = fftLength / 2 + 1;
        int32_t const transformStride = fftLength + 2;
        int64_t const rows = static_cast<int64_t>(batch) * numFrames;
        int64_t const normalizationBlocks = static_cast<int64_t>(batch) * numFeatures;
        WorkspaceLayout layout{};
        if (rows != mPlanRows || fftLength != mPlanLength
            || normalizationBlocks < 1
            || normalizationBlocks > std::numeric_limits<int32_t>::max()
            || !makeWorkspaceLayout(rows, fftLength, layout))
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
        auto* power = frames;
        auto* cublasWorkspace = workspaceBytes + layout.cublasOffset;

        // cuFFT and cuBLAS handles belong to this execution-context clone. Only
        // update their stream state when TensorRT changes the execution stream.
        if (!mStreamInitialized || stream != mStream)
        {
            if (cufftSetStream(mPlan, stream) != CUFFT_SUCCESS
                || cublasSetStream(mCublas, stream) != CUBLAS_STATUS_SUCCESS)
            {
                return 1;
            }
            mStream = stream;
            mStreamInitialized = true;
            // cublasSetStream resets the handle's workspace selection. Clear
            // the pointer cache so this execution reapplies TensorRT's region.
            mCublasWorkspace = nullptr;
        }

        prepareFrames<<<static_cast<uint32_t>(rows), kThreadsPerBlock, 0,
            stream>>>(static_cast<float const*>(inputs[0]),
            static_cast<int64_t const*>(inputs[1]),
            static_cast<float const*>(inputs[2]), frames, audioSamples,
            numFrames, fftLength, mParameters.frameShift,
            mParameters.preemph);
        if (cudaGetLastError() != cudaSuccess
            || cufftExecR2C(mPlan, frames, reinterpret_cast<cufftComplex*>(frames))
                != CUFFT_SUCCESS)
        {
            return 1;
        }

        int32_t const powerBlocks = static_cast<int32_t>(
            std::min<int64_t>(kMaxBlocks, rows));
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
        // Row-major power has the padded in-place FFT stride. Viewing it and mel
        // storage as transposed column-major matrices writes row-major output.
        cublasStatus_t const gemm = cublasGemmEx(mCublas, CUBLAS_OP_N,
            CUBLAS_OP_N, numFeatures, static_cast<int32_t>(rows), frequencies,
            &alpha, inputs[3], CUDA_R_32F, numFeatures, power, CUDA_R_32F,
            transformStride, &beta, outputs[0], CUDA_R_32F, numFeatures,
            getCublasComputeType(mTactic), CUBLAS_GEMM_DEFAULT);
        if (gemm != CUBLAS_STATUS_SUCCESS)
        {
            return 1;
        }

        if (batch >= mCoalescedNormalizationMinBatch)
        {
            normalizeFeaturesCoalesced<<<static_cast<uint32_t>(batch),
                kCoalescedNormalizationThreads, 0, stream>>>(
                static_cast<float*>(outputs[0]),
                static_cast<int64_t const*>(inputs[1]),
                static_cast<int32_t*>(outputs[1]), audioSamples, numFrames,
                numFeatures, mParameters.frameShift, mParameters.logEps,
                mParameters.eps);
        }
        else
        {
            normalizeFeaturesParallel<<<
                static_cast<uint32_t>(normalizationBlocks),
                kParallelNormalizationThreads, 0, stream>>>(
                static_cast<float*>(outputs[0]),
                static_cast<int64_t const*>(inputs[1]),
                static_cast<int32_t*>(outputs[1]), audioSamples, numFrames,
                numFeatures, mParameters.frameShift, mParameters.logEps,
                mParameters.eps);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(
        IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        // Give every execution context independent library handles, FFT plan,
        // tactic, and cached stream state so concurrent contexts cannot race.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    friend class ParakeetFeaturePluginCreator;

    void initializeFields() noexcept
    {
        mSerializedFields = {{
            {kFrameShiftField, &mParameters.frameShift, PluginFieldType::kINT32, 1},
            {kPreemphField, &mParameters.preemph, PluginFieldType::kFLOAT32, 1},
            {kLogEpsField, &mParameters.logEps, PluginFieldType::kFLOAT32, 1},
            {kEpsField, &mParameters.eps, PluginFieldType::kFLOAT32, 1},
        }};
        mFields = {static_cast<int32_t>(mSerializedFields.size()),
            mSerializedFields.data()};
    }

    FeatureParameters mParameters{};
    cublasHandle_t mCublas{nullptr};
    cufftHandle mPlan{};
    cudaStream_t mStream{nullptr};
    void* mCublasWorkspace{nullptr};
    int32_t mPlanRows{};
    int32_t mPlanLength{};
    int32_t mTactic{kStrictComputeTactic};
    int32_t mCoalescedNormalizationMinBatch{std::numeric_limits<int32_t>::max()};
    bool mInitialized{false};
    bool mStreamInitialized{false};
    std::array<char, kTimingCacheIdSize> mTimingCacheId{};
    std::array<PluginField, 4> mSerializedFields{};
    PluginFieldCollection mFields{};
};

class ParakeetFeaturePluginCreator final : public IPluginCreatorV3One
{
public:
    ParakeetFeaturePluginCreator() noexcept
    {
        mAttributes = {{
            {kFrameShiftField, nullptr, PluginFieldType::kINT32, 1},
            {kPreemphField, nullptr, PluginFieldType::kFLOAT32, 1},
            {kLogEpsField, nullptr, PluginFieldType::kFLOAT32, 1},
            {kEpsField, nullptr, PluginFieldType::kFLOAT32, 1},
        }};
        mFields = {
            static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
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

        FeatureParameters parameters{};
        uint32_t foundFields = 0;
        constexpr uint32_t allFields = (1U << 4) - 1;
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
            if (fieldName == kFrameShiftField)
            {
                fieldIndex = 0;
                expectedType = PluginFieldType::kINT32;
                destination = &parameters.frameShift;
            }
            else if (fieldName == kPreemphField)
            {
                fieldIndex = 1;
                expectedType = PluginFieldType::kFLOAT32;
                destination = &parameters.preemph;
            }
            else if (fieldName == kLogEpsField)
            {
                fieldIndex = 2;
                expectedType = PluginFieldType::kFLOAT32;
                destination = &parameters.logEps;
            }
            else if (fieldName == kEpsField)
            {
                fieldIndex = 3;
                expectedType = PluginFieldType::kFLOAT32;
                destination = &parameters.eps;
            }
            else
            {
                // TensorRT may append implementation metadata. Ignore unknown
                // fields while requiring every frontend parameter exactly once.
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

        auto* plugin = new (std::nothrow) ParakeetFeaturePlugin(parameters);
        if (plugin == nullptr || !plugin->mInitialized)
        {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

private:
    std::array<PluginField, 4> mAttributes{};
    PluginFieldCollection mFields{};
};
} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrParakeetFeaturePlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Runtime and builder have separate registries. Treat an existing creator
    // as success so repeated package imports remain harmless.
    static ParakeetFeaturePluginCreator runtimeCreator;
    static ParakeetFeaturePluginCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry,
                                ParakeetFeaturePluginCreator& creator) noexcept {
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
            || registry->getCreator(
                   kPluginName, kPluginVersion, kPluginNamespace)
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
