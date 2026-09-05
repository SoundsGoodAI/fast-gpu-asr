#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Compile and execute CPU-only contracts for the shared cuBLAS tactic header."""

import shutil
import subprocess
from pathlib import Path

import pytest
import tensorrt as trt
from cuda.pathfinder import find_nvidia_header_directory

PLUGIN_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "fast_gpu_asr" / "tensorrt_plugins"
)
TENSORRT_DATA_TYPES = {
    "FLOAT": 0,
    "HALF": 1,
    "INT8": 2,
    "INT32": 3,
    "BOOL": 4,
    "UINT8": 5,
    "FP8": 6,
    "BF16": 7,
    "INT64": 8,
    "INT4": 9,
    "FP4": 10,
    "E8M0": 11,
}


def compile_and_run(tmp_path: Path, sources: tuple[str, ...]) -> None:
    """Compile and execute a header contract without linking GPU libraries.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for source files, the enum stub, and the executable.
    sources : tuple[str, ...]
        C++20 translation units, exactly one of which defines ``main``.

    Notes
    -----
    The stub supplies only TensorRT's DataType enum, checked separately against
    its Python bindings. Real cuBLAS headers come from cuda-pathfinder. Missing
    development tools skip compilation; compiler errors and failed assertions
    propagate through subprocess checks, each with a 60-second timeout.
    """

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the cuBLAS tactic header test.")
    cuda_include = find_nvidia_header_directory("cublas")
    if cuda_include is None:
        pytest.skip("CUDA cuBLAS headers are required for the tactic header test.")

    members = "\n".join(
        f"    k{name} = {value}," for name, value in TENSORRT_DATA_TYPES.items()
    )
    (tmp_path / "NvInferRuntimeBase.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        "#define FAST_GPU_ASR_TEST_TENSORRT_STUB 1\n"
        "namespace nvinfer1 { enum class DataType : std::int32_t {\n"
        f"{members}\n"
        "}; }\n",
        encoding="utf8",
    )
    paths = [tmp_path / f"part_{index}.cpp" for index in range(len(sources))]
    for path, source in zip(paths, sources, strict=True):
        path.write_text(source, encoding="utf8")
    executable = tmp_path / "cublas_tactics_test"
    subprocess.run(
        (
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic-errors",
            "-I",
            str(PLUGIN_DIR),
            "-I",
            str(tmp_path),
            "-I",
            str(cuda_include),
            *(str(path) for path in paths),
            "-o",
            str(executable),
        ),
        check=True,
        timeout=60,
    )
    subprocess.run((str(executable),), check=True, timeout=60)


def test_tensorrt_data_type_stub_matches_installed_bindings() -> None:
    assert {
        name: int(value) for name, value in trt.DataType.__members__.items()
    } == TENSORRT_DATA_TYPES


def test_cublas_tactics_are_self_contained_and_dtype_safe(tmp_path: Path) -> None:
    compile_and_run(
        tmp_path,
        (
            r"""
#include "cublas_tactics.h"
#include "cublas_tactics.h"

#ifndef FAST_GPU_ASR_TEST_TENSORRT_STUB
#error "The contract test requires its isolated TensorRT stub."
#endif
#ifdef NDEBUG
#error "The contract test requires active assertions."
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
#include <utility>

using namespace fastgpuasr_tensorrt;
using nvinfer1::DataType;

int main()
{
    static_assert(kStrictComputeTactic == 1 && kFast16FComputeTactic == 2
        && kFast16BFComputeTactic == 3 && kFastTF32ComputeTactic == 4);
    static_assert(kCublasComputeTactics == std::array<int32_t, 4>{1, 2, 3, 4});
    static_assert(kReducedStorageCublasComputeTactics == std::array<int32_t, 1>{1});
    static_assert(std::is_same_v<decltype(getCublasComputeTactics()),
        std::span<int32_t const>>);
    static_assert(std::is_same_v<std::underlying_type_t<DataType>, int32_t>);
    static_assert(std::is_same_v<decltype(isCublasComputeTactic(1)), bool>);
    static_assert(std::is_same_v<decltype(getCublasComputeTacticCount()), int32_t>);
    static_assert(std::is_same_v<decltype(getCublasComputeType(1)),
        cublasComputeType_t>);
    static_assert(std::is_same_v<decltype(writeCublasComputeTactics(nullptr, 0)),
        int32_t>);
    static_assert(std::is_same_v<decltype(setCublasComputeTactic(
        1, std::declval<int32_t&>())), int32_t>);

    static_assert(std::ranges::equal(getCublasComputeTactics(),
        std::array<int32_t, 4>{1, 2, 3, 4}));
    static_assert(std::ranges::equal(getCublasComputeTactics(DataType::kHALF),
        std::array<int32_t, 1>{1}));
    static_assert(std::ranges::equal(getCublasComputeTactics(DataType::kBF16),
        std::array<int32_t, 1>{1}));
    static_assert(getCublasComputeTactics(DataType::kINT8).empty());
    static_assert(isCublasComputeTactic(4) && !isCublasComputeTactic(0));
    static_assert(isCublasComputeTactic(1, DataType::kHALF)
        && !isCublasComputeTactic(2, DataType::kHALF));
    static_assert(getCublasComputeTacticCount() == 4);
    static_assert(getCublasComputeTacticCount(DataType::kHALF) == 1);
    static_assert(getCublasComputeTacticCount(DataType::kBF16) == 1);
    static_assert(getCublasComputeTacticCount(DataType::kINT8) == 0);
    static_assert(getCublasComputeType(1) == CUBLAS_COMPUTE_32F);
    static_assert(getCublasComputeType(2) == CUBLAS_COMPUTE_32F_FAST_16F);
    static_assert(getCublasComputeType(3) == CUBLAS_COMPUTE_32F_FAST_16BF);
    static_assert(getCublasComputeType(4) == CUBLAS_COMPUTE_32F_FAST_TF32);
    static_assert(getCublasComputeType(0) == CUBLAS_COMPUTE_32F);
    static_assert(noexcept(getCublasComputeTactics()));
    static_assert(noexcept(isCublasComputeTactic(1)));
    static_assert(noexcept(getCublasComputeTacticCount()));
    static_assert(noexcept(getCublasComputeType(1)));
    static_assert(noexcept(writeCublasComputeTactics(nullptr, 0)));
    static_assert(noexcept(setCublasComputeTactic(1, std::declval<int32_t&>())));

    for (DataType const type : {
             DataType::kFLOAT, DataType::kHALF, DataType::kBF16,
             DataType::kINT8, DataType::kINT32, DataType::kBOOL,
             DataType::kUINT8, DataType::kFP8, DataType::kINT64,
             DataType::kINT4, DataType::kFP4, DataType::kE8M0,
             static_cast<DataType>(-1), static_cast<DataType>(INT32_MIN),
             static_cast<DataType>(INT32_MAX)})
    {
        int32_t const count = type == DataType::kFLOAT ? 4
            : (type == DataType::kHALF || type == DataType::kBF16) ? 1 : 0;
        std::array<int32_t, 4> const expectedTactics{1, 2, 3, 4};
        auto const tactics = getCublasComputeTactics(type);
        assert(getCublasComputeTacticCount(type) == count);
        assert(std::equal(tactics.begin(), tactics.end(),
            expectedTactics.begin(), expectedTactics.begin() + count));

        for (int32_t tactic : {INT32_MIN, -1, 0, 1, 2, 3, 4, 5, INT32_MAX})
        {
            bool const valid = tactic >= 1 && tactic <= count;
            assert(isCublasComputeTactic(tactic, type) == valid);
            bool const accepted = valid || (tactic == 0 && count > 0);
            int32_t selected = 71;
            assert(setCublasComputeTactic(tactic, selected, type)
                == (accepted ? 0 : 1));
            assert(selected == (accepted ? std::max(1, tactic) : 71));
        }

        for (int32_t requestedCount :
             {INT32_MIN, -1, 0, 1, 2, 3, 4, 5, INT32_MAX})
        {
            bool const accepted = count > 0 && requestedCount == count;
            std::array<int32_t, 6> guarded{91, 92, 93, 94, 95, 96};
            auto expected = guarded;
            if (accepted)
            {
                std::copy_n(expectedTactics.begin(), count, expected.begin() + 1);
            }
            assert(writeCublasComputeTactics(
                       guarded.data() + 1, requestedCount, type)
                == (accepted ? 0 : 1));
            assert(guarded == expected);
            assert(writeCublasComputeTactics(nullptr, requestedCount, type) == 1);
        }
    }

    std::array<int32_t, 4> written{};
    assert(writeCublasComputeTactics(written.data(), 4) == 0);
    assert((written == std::array<int32_t, 4>{1, 2, 3, 4}));
    for (int32_t tactic : {0, 1, 2, 3, 4})
    {
        int32_t selected = 71;
        assert(setCublasComputeTactic(tactic, selected) == 0);
        assert(selected == std::max(1, tactic));
    }
    for (int32_t invalid : {INT32_MIN, -1, 0, 5, INT32_MAX})
    {
        assert(!isCublasComputeTactic(invalid));
        assert(getCublasComputeType(invalid) == CUBLAS_COMPUTE_32F);
    }
}
""",
        ),
    )


def test_cublas_tactics_support_turing(tmp_path: Path) -> None:
    compile_and_run(
        tmp_path,
        (
            r"""
#include "cublas_tactics.h"
#include <algorithm>
#include <cassert>

using namespace fastgpuasr_tensorrt;
using nvinfer1::DataType;

int deviceMajor = 7;
cudaError_t deviceStatus = cudaSuccess;
cudaError_t attributeStatus = cudaSuccess;

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device)
{
    *device = 3;
    return deviceStatus;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(
    int* value, cudaDeviceAttr attribute, int device)
{
    assert(device == 3 && attribute == cudaDevAttrComputeCapabilityMajor);
    *value = deviceMajor;
    return attributeStatus;
}

int main()
{
    for (int major : {7, 8, 9, 12})
    {
        deviceMajor = major;
        bool const ampere = deviceSupportsAmpereCompute();
        assert(ampere == (major >= 8));
        for (DataType type : {DataType::kFLOAT, DataType::kHALF, DataType::kBF16})
        {
            int const count = type == DataType::kFLOAT ? (ampere ? 4 : 2)
                : type == DataType::kHALF || ampere ? 1 : 0;
            auto const tactics = getCublasComputeTactics(type, ampere);
            assert(getCublasComputeTacticCount(type, ampere) == count);
            assert(static_cast<int>(tactics.size()) == count);
            for (int i = 0; i < count; ++i)
            {
                assert(tactics[i] == i + 1);
            }
            for (int requested : {-1, 0, 1, 2, 3, 4, 5})
            {
                std::array<int32_t, 6> guarded{91, 92, 93, 94, 95, 96};
                auto expected = guarded;
                bool const accepted = count > 0 && requested == count;
                if (accepted)
                {
                    std::copy(tactics.begin(), tactics.end(), expected.begin() + 1);
                }
                assert(writeCublasComputeTactics(
                    guarded.data() + 1, requested, type, ampere) == !accepted);
                assert(guarded == expected);
            }
        }
    }
    deviceStatus = cudaErrorInvalidDevice;
    assert(!deviceSupportsAmpereCompute());
    deviceStatus = cudaSuccess;
    attributeStatus = cudaErrorInvalidValue;
    assert(!deviceSupportsAmpereCompute());
}
""",
        ),
    )


def test_cublas_tactics_header_links_across_translation_units(tmp_path: Path) -> None:
    compile_and_run(
        tmp_path,
        (
            r"""
#include "cublas_tactics.h"

int32_t firstTacticCount()
{
    return fastgpuasr_tensorrt::getCublasComputeTacticCount();
}
""",
            r"""
#include "cublas_tactics.h"
#include <cassert>

#ifdef NDEBUG
#error "The link test requires active assertions."
#endif

int32_t firstTacticCount();

int main()
{
    assert(firstTacticCount() == 4);
    assert(fastgpuasr_tensorrt::isCublasComputeTactic(
        fastgpuasr_tensorrt::kStrictComputeTactic, nvinfer1::DataType::kHALF));
}
""",
        ),
    )
