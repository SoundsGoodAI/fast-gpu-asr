// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "plugin_namespace.h"

namespace fastgpuasr_tensorrt
{
using namespace nvinfer1;

namespace
{

// All six inputs are contiguous NTC outputs from consecutive encoder stacks.
// For stack channel counts C4 >= C5 >= C6, the result is
// concat(stack6, stack5[..., C6:C5], stack4[..., C5:C4]). Inputs 0-2 are
// retained as opaque dependencies to prevent unsafe cross-stack fusion.
constexpr char const* kPluginName = "zipformer_output_assembly";
constexpr char const* kPluginVersion = "1";
constexpr int32_t kInputCount = 6;
constexpr int32_t kOutputCount = 1;
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kVectorBytes = sizeof(uint4);
static_assert(sizeof(uint4) == 16);
static_assert(alignof(uint4) == 16);
// The grid-stride loop covers larger tensors; this cap already provides enough
// resident blocks to saturate supported GPUs without an excessive launch grid.
constexpr int64_t kMaxBlocks = 65535;

constexpr bool isSupportedDataType(DataType type) noexcept
{
    return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

constexpr int32_t dataTypeBytes(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kFLOAT: return sizeof(float);
    case DataType::kHALF:
    case DataType::kBF16: return sizeof(uint16_t);
    default: return 0;
    }
}

bool hasAddressableBytes(Dims const& dims, int32_t elementBytes) noexcept
{
    // Kernel offsets use signed 64-bit vector indexes, while pointer arithmetic
    // ultimately addresses bytes. Prove both products fit before launching.
    if (dims.nbDims < 1 || elementBytes < 1)
    {
        return false;
    }

    int64_t elements = 1;
    int64_t const maxElements = std::numeric_limits<int64_t>::max() / elementBytes;
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (dims.d[index] < 1 || dims.d[index] > std::numeric_limits<int32_t>::max()
            || elements > maxElements / dims.d[index])
        {
            return false;
        }
        elements *= dims.d[index];
    }
    return true;
}

bool haveValidShapes(Dims const* inputs, Dims const& output, DataType type) noexcept
{
    int32_t const elementBytes = dataTypeBytes(type);
    if (inputs == nullptr || output.nbDims != 3 || !hasAddressableBytes(output, elementBytes))
    {
        return false;
    }

    // Every retained stack output must describe the same N/T grid. Inputs 0-2
    // contribute only graph dependencies, so their channel counts need no
    // relationship to the three channel bands assembled from inputs 3-5.
    for (int32_t index = 0; index < kInputCount; ++index)
    {
        if (inputs[index].nbDims != 3 || !hasAddressableBytes(inputs[index], elementBytes)
            || inputs[index].d[0] != inputs[0].d[0] || inputs[index].d[1] != inputs[0].d[1])
        {
            return false;
        }
    }

    if (output.d[0] != inputs[3].d[0] || output.d[1] != inputs[3].d[1]
        || output.d[2] != inputs[3].d[2])
    {
        return false;
    }

    // Requiring every band boundary to fall on 16 bytes lets the kernel copy
    // complete uint4 values without scalar prologues or tails. Equal adjacent
    // widths are valid and simply make the corresponding channel band empty.
    int64_t const encoder4Bytes = static_cast<int64_t>(inputs[3].d[2]) * elementBytes;
    int64_t const encoder5Bytes = static_cast<int64_t>(inputs[4].d[2]) * elementBytes;
    int64_t const encoder6Bytes = static_cast<int64_t>(inputs[5].d[2]) * elementBytes;
    return inputs[3].d[2] >= inputs[4].d[2] && inputs[4].d[2] >= inputs[5].d[2]
           && encoder4Bytes % kVectorBytes == 0 && encoder5Bytes % kVectorBytes == 0
           && encoder6Bytes % kVectorBytes == 0;
}

// Inputs 0-2 deliberately remain plugin dependencies even though the final
// representation only contains channel bands from inputs 3-5. This opaque
// boundary prevents unsafe TensorRT fusion across the six encoder stacks.
// Each thread moves one aligned 16-byte vector from the stack that owns the
// corresponding output channel band.
__global__ void assembleOutput(uint4 const* __restrict__ encoder4,
    uint4 const* __restrict__ encoder5, uint4 const* __restrict__ encoder6,
    uint4* __restrict__ output, int64_t totalVectors, int32_t outputVectors,
    int32_t encoder5Vectors, int32_t encoder6Vectors)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < totalVectors; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        // Flattening N and T is safe because every NTC row has the same
        // aligned channel-band layout.
        int32_t const channel = index % outputVectors;
        int64_t const frame = index / outputVectors;
        // Source and destination retain the same channel coordinate. Only the
        // source row stride changes as the encoder width narrows.
        if (channel < encoder6Vectors)
        {
            output[index] = encoder6[frame * encoder6Vectors + channel];
        }
        else if (channel < encoder5Vectors)
        {
            output[index] = encoder5[frame * encoder5Vectors + channel];
        }
        else
        {
            // Encoder 4 and the output have identical row widths, so their
            // flattened vector indexes are the same for the final band.
            output[index] = encoder4[index];
        }
    }
}

// Assemble the surviving Zipformer channel bands into one NTC tensor while
// retaining all six encoder-stack outputs as opaque graph dependencies.
class OutputAssemblyPlugin final : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
{
  public:
    OutputAssemblyPlugin() noexcept { mFields = {0, nullptr}; }

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

    IPluginV3* clone() noexcept override { return new (std::nothrow) OutputAssemblyPlugin(); }

    char const* getPluginName() const noexcept override { return kPluginName; }

    char const* getPluginVersion() const noexcept override { return kPluginVersion; }

    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    int32_t getNbOutputs() const noexcept override { return kOutputCount; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || outputs[0].desc.dims.nbDims != 3
            || !isSupportedDataType(inputs[0].desc.type)
            || outputs[0].desc.type != inputs[0].desc.type
            || outputs[0].desc.format != TensorFormat::kLINEAR)
        {
            return 1;
        }
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            if (inputs[index].desc.dims.nbDims != 3
                || inputs[index].desc.type != inputs[0].desc.type
                || inputs[index].desc.format != TensorFormat::kLINEAR)
            {
                return 1;
            }
        }

        // Build-time descriptors contain a dynamic profile rather than one
        // concrete shape. Validate its min/opt/max points here;
        // onShapeChange() validates each actual shape before enqueue().
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
                       && haveValidShapes(optInputs, outputs[0].opt, inputs[0].desc.type)
                       && haveValidShapes(maxInputs, outputs[0].max, inputs[0].desc.type)
                   ? 0
                   : 1;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !isSupportedDataType(inputTypes[0]))
        {
            return 1;
        }
        for (int32_t index = 1; index < kInputCount; ++index)
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
            || nbOutputs != kOutputCount || nbShapeInputs != 0)
        {
            return 1;
        }
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            if (inputs[index].nbDims != 3 || inputs[index].d[0] == nullptr
                || inputs[index].d[1] == nullptr || inputs[index].d[2] == nullptr)
            {
                return 1;
            }
        }
        outputs[0] = inputs[3];
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
        return desc.format == TensorFormat::kLINEAR && isSupportedDataType(desc.type)
               && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
        int32_t) const noexcept override
    {
        return 0;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount
            || nbOutputs != kOutputCount || !isSupportedDataType(inputs[0].type)
            || outputs[0].type != inputs[0].type || outputs[0].format != TensorFormat::kLINEAR)
        {
            return 1;
        }

        Dims inputShapes[kInputCount]{};
        for (int32_t index = 0; index < kInputCount; ++index)
        {
            if (inputs[index].type != inputs[0].type
                || inputs[index].format != TensorFormat::kLINEAR)
            {
                return 1;
            }
            inputShapes[index] = inputs[index].dims;
        }
        return haveValidShapes(inputShapes, outputs[0].dims, inputs[0].type) ? 0 : 1;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        static_cast<void>(workspace);
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || outputs[0] == nullptr)
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
        if (onShapeChange(inputDesc, kInputCount, outputDesc, kOutputCount) != 0)
        {
            return 1;
        }

        // The vectorized kernel requires aligned bases and aligned NTC row
        // strides. Row alignment is established by haveValidShapes().
        auto isAligned = [](void const* pointer) noexcept
        { return reinterpret_cast<uintptr_t>(pointer) % alignof(uint4) == 0; };
        if (!isAligned(inputs[3]) || !isAligned(inputs[4]) || !isAligned(inputs[5])
            || !isAligned(outputs[0]))
        {
            return 1;
        }

        int32_t const elementBytes = dataTypeBytes(inputDesc[0].type);
        int64_t const outputBytes = static_cast<int64_t>(inputDesc[3].dims.d[2]) * elementBytes;
        int64_t const encoder5Bytes = static_cast<int64_t>(inputDesc[4].dims.d[2]) * elementBytes;
        int64_t const encoder6Bytes = static_cast<int64_t>(inputDesc[5].dims.d[2]) * elementBytes;
        int64_t const outputVectorCount = outputBytes / kVectorBytes;
        int64_t const encoder5VectorCount = encoder5Bytes / kVectorBytes;
        int64_t const encoder6VectorCount = encoder6Bytes / kVectorBytes;
        if (outputVectorCount < 1 || outputVectorCount > std::numeric_limits<int32_t>::max()
            || encoder5VectorCount < 1 || encoder5VectorCount > std::numeric_limits<int32_t>::max()
            || encoder6VectorCount < 1 || encoder6VectorCount > std::numeric_limits<int32_t>::max())
        {
            return 1;
        }
        // The kernel keeps per-row vector counts in int32_t. Narrow only after
        // checking the bounds locally rather than relying on profile validation.
        int32_t const outputVectors = static_cast<int32_t>(outputVectorCount);
        int32_t const encoder5Vectors = static_cast<int32_t>(encoder5VectorCount);
        int32_t const encoder6Vectors = static_cast<int32_t>(encoder6VectorCount);

        // Batch and time do not affect the band mapping, so launch one
        // grid-stride copy over all 16-byte vectors in all NTC rows.
        int64_t const frames =
            static_cast<int64_t>(inputDesc[3].dims.d[0]) * inputDesc[3].dims.d[1];
        if (frames > std::numeric_limits<int64_t>::max() / outputVectors)
        {
            return 1;
        }
        int64_t const totalVectors = frames * outputVectors;
        int64_t const requiredBlocks = (totalVectors - 1) / kThreadsPerBlock + 1;
        int32_t const blocks = static_cast<int32_t>(std::min(requiredBlocks, kMaxBlocks));

        // Preserve errors from earlier asynchronous work instead of silently
        // attributing success to this invocation.
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return 1;
        }

        assembleOutput<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<uint4 const*>(inputs[3]), static_cast<uint4 const*>(inputs[4]),
            static_cast<uint4 const*>(inputs[5]), static_cast<uint4*>(outputs[0]), totalVectors,
            outputVectors, encoder5Vectors, encoder6Vectors);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        static_cast<void>(context);
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFields; }

  private:
    PluginFieldCollection mFields{};
};

class OutputAssemblyCreator final : public IPluginCreatorV3One
{
  public:
    OutputAssemblyCreator() noexcept { mFields = {0, nullptr}; }

    char const* getPluginName() const noexcept override { return kPluginName; }

    char const* getPluginVersion() const noexcept override { return kPluginVersion; }

    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fields,
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
        return new (std::nothrow) OutputAssemblyPlugin();
    }

  private:
    PluginFieldCollection mFields{};
};

} // namespace
} // namespace fastgpuasr_tensorrt

extern "C" bool initFastGpuAsrZipformerOutputAssemblyPlugin() noexcept
{
    using namespace fastgpuasr_tensorrt;

    // Builder and runtime use distinct registries. Treat an existing matching
    // creator as success so repeated package imports remain idempotent.
    static OutputAssemblyCreator runtimeCreator;
    static OutputAssemblyCreator builderCreator;
    auto ensureRegistered = [](IPluginRegistry* registry, OutputAssemblyCreator& creator) noexcept
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
