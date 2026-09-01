#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Compile-level tests for shared cuBLAS plugin tactic helpers."""

import shutil
import subprocess
from pathlib import Path

import pytest

cuda_pathfinder = pytest.importorskip("cuda.pathfinder")

PLUGIN_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "fast_gpu_asr" / "tensorrt_plugins"
)
TENSORRT_DATA_TYPE_STUB = r"""
#pragma once

namespace nvinfer1
{
enum class DataType
{
    kFLOAT,
    kHALF,
    kINT8,
    kINT32,
    kBOOL,
    kUINT8,
    kFP8,
    kBF16,
    kINT64,
    kINT4,
    kFP4,
    kE8M0,
};
}
"""


def cxx_compile_prefix(tmp_path: Path) -> tuple[str, ...]:
    """Prepare the isolated C++20 include environment used by header tests."""

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the cuBLAS tactic header test.")
    cuda_include = cuda_pathfinder.find_nvidia_header_directory("cublas")
    if cuda_include is None:
        pytest.skip("CUDA cuBLAS headers are required for the tactic header test.")

    (tmp_path / "NvInferRuntimeBase.h").write_text(
        TENSORRT_DATA_TYPE_STUB, encoding="utf8"
    )
    return (
        compiler,
        "-std=c++20",
        "-I",
        str(PLUGIN_DIR),
        "-I",
        str(tmp_path),
        "-I",
        str(cuda_include),
    )


def test_cublas_tactics_are_self_contained_and_dtype_safe(tmp_path: Path) -> None:
    """Compile the header alone and verify its public tactic contract."""

    compile_prefix = cxx_compile_prefix(tmp_path)
    source = tmp_path / "cublas_tactics_test.cpp"
    executable = tmp_path / "cublas_tactics_test"
    source.write_text(
        r"""
#include "cublas_tactics.h"
#include "cublas_tactics.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <type_traits>
#include <utility>

#ifdef NDEBUG
#error "The cuBLAS tactic contract test requires active assertions."
#endif

using namespace fastgpuasr_tensorrt;
using nvinfer1::DataType;

int main()
{
    static_assert(kStrictComputeTactic == 1);
    static_assert(kFast16FComputeTactic == 2);
    static_assert(kFast16BFComputeTactic == 3);
    static_assert(kFastTF32ComputeTactic == 4);
    static_assert(kCublasComputeTactics
        == std::array<int32_t, 4>{1, 2, 3, 4});
    static_assert(std::is_same_v<decltype(getCublasComputeTactics()),
        std::span<int32_t const>>);
    static_assert(noexcept(getCublasComputeTactics()));
    static_assert(noexcept(isCublasComputeTactic(1)));
    static_assert(noexcept(getCublasComputeTacticCount()));
    static_assert(noexcept(getCublasComputeType(1)));
    static_assert(noexcept(writeCublasComputeTactics(nullptr, 0)));
    static_assert(noexcept(setCublasComputeTactic(
        1, std::declval<int32_t&>())));

    auto const floatTactics = getCublasComputeTactics();
    assert(floatTactics.size() == kCublasComputeTactics.size());
    assert(std::equal(floatTactics.begin(), floatTactics.end(),
        kCublasComputeTactics.begin()));
    assert(getCublasComputeTacticCount(DataType::kFLOAT) == 4);
    assert(getCublasComputeTacticCount(DataType::kHALF) == 1);
    assert(getCublasComputeTacticCount(DataType::kBF16) == 1);
    for (DataType const unsupported : {
             DataType::kINT8,
             DataType::kINT32,
             DataType::kBOOL,
             DataType::kUINT8,
             DataType::kFP8,
             DataType::kINT64,
             DataType::kINT4,
             DataType::kFP4,
             DataType::kE8M0,
             static_cast<DataType>(-1),
             static_cast<DataType>(INT32_MAX),
         })
    {
        assert(getCublasComputeTactics(unsupported).empty());
        assert(getCublasComputeTacticCount(unsupported) == 0);
        for (int32_t tactic : {0, 1, 2, 3, 4, 5, INT32_MIN, INT32_MAX})
        {
            assert(!isCublasComputeTactic(tactic, unsupported));
            int32_t selected = 71;
            assert(setCublasComputeTactic(tactic, selected, unsupported) == 1);
            assert(selected == 71);
        }
        for (int32_t count : {0, 1})
        {
            std::array<int32_t, 1> unchanged{73};
            assert(writeCublasComputeTactics(
                       unchanged.data(), count, unsupported)
                == 1);
            assert(unchanged[0] == 73);
        }
    }

    for (int32_t tactic : kCublasComputeTactics)
    {
        assert(isCublasComputeTactic(tactic));
    }
    for (DataType const reducedType : {DataType::kHALF, DataType::kBF16})
    {
        for (int32_t tactic : kCublasComputeTactics)
        {
            assert(isCublasComputeTactic(tactic, reducedType)
                == (tactic == kStrictComputeTactic));
        }
    }
    for (int32_t invalid : {0, -1, 5, INT32_MIN, INT32_MAX})
    {
        assert(!isCublasComputeTactic(invalid));
    }

    std::array<int32_t, 6> tactics{
        91, 92, 93, 94, 95, 96,
    };
    assert(writeCublasComputeTactics(
               tactics.data() + 1, 4, DataType::kFLOAT)
        == 0);
    assert(tactics.front() == 91);
    assert(tactics.back() == 96);
    assert(std::equal(
        tactics.begin() + 1, tactics.end() - 1,
        kCublasComputeTactics.begin()));
    assert(writeCublasComputeTactics(nullptr, 4, DataType::kFLOAT) == 1);
    for (int32_t count : {
             INT32_MIN, -1, 0, 1, 3, 5, INT32_MAX})
    {
        std::array<int32_t, 4> unchanged{21, 22, 23, 24};
        assert(writeCublasComputeTactics(
                   unchanged.data(), count, DataType::kFLOAT)
            == 1);
        assert((unchanged == std::array<int32_t, 4>{21, 22, 23, 24}));
    }
    std::array<int32_t, 1> reduced{-1};
    assert(writeCublasComputeTactics(
               reduced.data(), 1, DataType::kHALF)
        == 0);
    assert(reduced[0] == kStrictComputeTactic);
    reduced[0] = -1;
    assert(writeCublasComputeTactics(
               reduced.data(), 1, DataType::kBF16)
        == 0);
    assert(reduced[0] == kStrictComputeTactic);
    assert(writeCublasComputeTactics(
               reduced.data(), 1, DataType::kINT8)
        == 1);

    for (DataType const reducedType : {DataType::kHALF, DataType::kBF16})
    {
        assert(writeCublasComputeTactics(nullptr, 1, reducedType) == 1);
        for (int32_t count : {INT32_MIN, -1, 0, 2, INT32_MAX})
        {
            std::array<int32_t, 2> unchanged{31, 32};
            assert(writeCublasComputeTactics(
                       unchanged.data(), count, reducedType)
                == 1);
            assert((unchanged == std::array<int32_t, 2>{31, 32}));
        }
    }

    for (DataType const supportedType : {
             DataType::kFLOAT, DataType::kHALF, DataType::kBF16})
    {
        for (int32_t tactic : getCublasComputeTactics(supportedType))
        {
            int32_t selected = 71;
            assert(setCublasComputeTactic(
                       tactic, selected, supportedType)
                == 0);
            assert(selected == tactic);
        }
        int32_t selected = 71;
        assert(setCublasComputeTactic(0, selected, supportedType) == 0);
        assert(selected == kStrictComputeTactic);
    }
    for (DataType const reducedType : {DataType::kHALF, DataType::kBF16})
    {
        for (int32_t tactic : {
                 kFast16FComputeTactic,
                 kFast16BFComputeTactic,
                 kFastTF32ComputeTactic,
             })
        {
            int32_t selected = 71;
            assert(setCublasComputeTactic(tactic, selected, reducedType) == 1);
            assert(selected == 71);
        }
    }
    for (int32_t invalid : {-1, 5, INT32_MIN, INT32_MAX})
    {
        int32_t selected = 71;
        assert(setCublasComputeTactic(invalid, selected) == 1);
        assert(selected == 71);
    }

    assert(getCublasComputeType(kStrictComputeTactic) == CUBLAS_COMPUTE_32F);
    assert(getCublasComputeType(kFast16FComputeTactic)
        == CUBLAS_COMPUTE_32F_FAST_16F);
    assert(getCublasComputeType(kFast16BFComputeTactic)
        == CUBLAS_COMPUTE_32F_FAST_16BF);
    assert(getCublasComputeType(kFastTF32ComputeTactic)
        == CUBLAS_COMPUTE_32F_FAST_TF32);
    for (int32_t invalid : {0, -1, 5, INT32_MIN, INT32_MAX})
    {
        assert(getCublasComputeType(invalid) == CUBLAS_COMPUTE_32F);
    }
}
""",
        encoding="utf8",
    )
    subprocess.run(
        (
            *compile_prefix,
            str(source),
            "-o",
            str(executable),
        ),
        check=True,
    )
    subprocess.run((str(executable),), check=True)


def test_cublas_tactics_header_links_across_translation_units(tmp_path: Path) -> None:
    """Compile and link independent translation units that include the header."""

    compile_prefix = cxx_compile_prefix(tmp_path)
    (tmp_path / "first.cpp").write_text(
        r"""
#include "cublas_tactics.h"

extern "C" int32_t firstTacticCount()
{
    return fastgpuasr_tensorrt::getCublasComputeTacticCount();
}
""",
        encoding="utf8",
    )
    (tmp_path / "second.cpp").write_text(
        r"""
#include "cublas_tactics.h"

extern "C" bool secondAcceptsStrictHalfTactic()
{
    return fastgpuasr_tensorrt::isCublasComputeTactic(
        fastgpuasr_tensorrt::kStrictComputeTactic,
        nvinfer1::DataType::kHALF);
}
""",
        encoding="utf8",
    )
    (tmp_path / "main.cpp").write_text(
        r"""
#include "cublas_tactics.h"

#include <cassert>

#ifdef NDEBUG
#error "The cuBLAS tactic link test requires active assertions."
#endif

extern "C" int32_t firstTacticCount();
extern "C" bool secondAcceptsStrictHalfTactic();

int main()
{
    assert(firstTacticCount() == 4);
    assert(secondAcceptsStrictHalfTactic());
}
""",
        encoding="utf8",
    )
    executable = tmp_path / "cublas_tactics_multi_tu"
    subprocess.run(
        (
            *compile_prefix,
            str(tmp_path / "first.cpp"),
            str(tmp_path / "second.cpp"),
            str(tmp_path / "main.cpp"),
            "-o",
            str(executable),
        ),
        check=True,
    )
    subprocess.run((str(executable),), check=True)
