#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Consistency tests for package and TensorRT-plugin constants."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from fast_gpu_asr.constants import (
    AUDIO_SAMPLES_PER_WORKER,
    DECODER_TYPES,
    INT32_MAX,
    MODEL_CONFIG_FILE,
    MODEL_TYPE_PARAKEET,
    MODEL_TYPE_ZIPFORMER,
    ONNX_OPSET_VERSION,
    PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
    PARAKEET_DECODER_ONNX_FILE,
    PARAKEET_DECODER_TENSORRT_FILE,
    PARAKEET_FEATURE_PLUGIN_NAME,
    PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
    PARAKEET_MAX_ENCODER_FRAMES,
    PARAKEET_ONNX_FILE,
    PARAKEET_TENSORRT_FILE,
    PRECISION_DTYPES,
    TDT_SEARCH_CHUNK_STEPS,
    TENSORRT_PLUGIN_NAMESPACE,
    TOKENIZER_FILE,
    TRANSDUCER_DECODER_TYPES,
    ZERO_LOG,
    ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
    ZIPFORMER_BEAM_SEARCH_THREADS,
    ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
    ZIPFORMER_DECODER_CONTEXTS_FILE,
    ZIPFORMER_DECODER_ONNX_FILE,
    ZIPFORMER_DECODER_TENSORRT_FILE,
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_FEATURE_PLUGIN_NAME,
    ZIPFORMER_ONNX_FILE,
    ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
    ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
    ZIPFORMER_TENSORRT_FILE,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
from fast_gpu_asr.tensorrt_plugins.constants import (
    CUDA_ARCHITECTURE_OPTIONS,
    CUDA_BUILD_LIBRARIES,
    CUDA_RUNTIME_LIBRARIES,
    NVCC_OPTIONS,
    PLUGIN_BUILDS,
    PLUGIN_INITIALIZERS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPOSITORY_ROOT / "src" / "fast_gpu_asr" / "tensorrt_plugins"
PLUGIN_NAMES_BY_SOURCE = {
    "zipformer_attention_value_plugin.cu": (ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,),
    "zipformer_relative_attention_plugin.cu": (
        ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
    ),
    "zipformer_convolution_plugin.cu": (ZIPFORMER_CONVOLUTION_PLUGIN_NAME,),
    "zipformer_feature_plugin.cu": (ZIPFORMER_FEATURE_PLUGIN_NAME,),
    "zipformer_resampling_plugin.cu": (
        ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
        ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
    ),
    "zipformer_output_assembly_plugin.cu": (ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,),
    "parakeet_feature_plugin.cu": (PARAKEET_FEATURE_PLUGIN_NAME,),
    "parakeet_flash_attention_plugin.cu": (PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,),
    "parakeet_convolution_plugin.cu": (PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,),
}


def test_precision_dtypes_cover_supported_export_precisions() -> None:
    assert {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    } == PRECISION_DTYPES


def test_supported_decoder_types() -> None:
    assert DECODER_TYPES == (
        "transducer_modified_beam_search",
        "transducer_greedy_search",
        "ctc_greedy_search",
    )
    assert TRANSDUCER_DECODER_TYPES == (
        "transducer_modified_beam_search",
        "transducer_greedy_search",
    )


def test_serialized_model_contract_constants_are_stable() -> None:
    assert (MODEL_TYPE_PARAKEET, MODEL_TYPE_ZIPFORMER) == (
        "parakeet_asr",
        "zipformer_asr",
    )
    assert (
        MODEL_CONFIG_FILE,
        TOKENIZER_FILE,
        PARAKEET_ONNX_FILE,
        PARAKEET_DECODER_ONNX_FILE,
        PARAKEET_TENSORRT_FILE,
        PARAKEET_DECODER_TENSORRT_FILE,
        ZIPFORMER_ONNX_FILE,
        ZIPFORMER_DECODER_ONNX_FILE,
        ZIPFORMER_TENSORRT_FILE,
        ZIPFORMER_DECODER_TENSORRT_FILE,
        ZIPFORMER_DECODER_CONTEXTS_FILE,
    ) == (
        "model_config.yaml",
        "bpe.model",
        "parakeet.onnx",
        "tdt_decoder.onnx",
        "parakeet.trt",
        "tdt_decoder.trt",
        "zipformer.onnx",
        "decoder.onnx",
        "zipformer.trt",
        "decoder.trt",
        "decoder_contexts.pt",
    )


def test_shared_scalar_constants_are_stable() -> None:
    assert type(ONNX_OPSET_VERSION) is int
    assert ONNX_OPSET_VERSION == 20
    assert TENSORRT_PLUGIN_NAMESPACE == "fast_gpu_asr"
    assert type(INT32_MAX) is int
    assert INT32_MAX == 2_147_483_647
    assert ZERO_LOG == -20.7233


def test_runtime_limits_and_tuning_values_satisfy_algorithm_constraints() -> None:
    assert type(AUDIO_SAMPLES_PER_WORKER) is int
    assert AUDIO_SAMPLES_PER_WORKER > 0
    assert type(PARAKEET_MAX_ENCODER_FRAMES) is int
    assert 0 < PARAKEET_MAX_ENCODER_FRAMES <= INT32_MAX
    assert type(TDT_SEARCH_CHUNK_STEPS) is int
    assert TDT_SEARCH_CHUNK_STEPS > 0
    assert TDT_SEARCH_CHUNK_STEPS % 2 == 0
    assert type(ZIPFORMER_BEAM_SEARCH_THREADS) is int
    assert 0 < ZIPFORMER_BEAM_SEARCH_THREADS <= 1024
    assert ZIPFORMER_BEAM_SEARCH_THREADS % 32 == 0


def test_plugin_identifiers_are_stable() -> None:
    plugin_names = tuple(
        plugin_name
        for source_plugin_names in PLUGIN_NAMES_BY_SOURCE.values()
        for plugin_name in source_plugin_names
    )

    assert plugin_names == (
        "zipformer_attention_value",
        "zipformer_relative_attention",
        "zipformer_convolution",
        "zipformer_feature_extractor",
        "zipformer_downsample",
        "zipformer_upsample_bypass",
        "zipformer_output_assembly",
        "parakeet_feature_extractor",
        "parakeet_flash_attention",
        "parakeet_conformer_convolution",
    )


def test_plugin_manifest_matches_native_sources() -> None:
    source_names = {path.name for path in PLUGIN_DIR.glob("*.cu")}
    manifest_sources = {source_name for source_name, _ in PLUGIN_BUILDS}
    initializers = dict(PLUGIN_INITIALIZERS)

    assert manifest_sources == source_names == set(PLUGIN_NAMES_BY_SOURCE)
    assert len(manifest_sources) == len(PLUGIN_BUILDS)
    assert set(initializers) == {
        Path(source_name).with_suffix(".so").name for source_name in source_names
    }
    assert (
        len(initializers) == len(set(initializers.values())) == len(PLUGIN_INITIALIZERS)
    )

    for source_name, plugin_names in PLUGIN_NAMES_BY_SOURCE.items():
        library_name = Path(source_name).with_suffix(".so").name
        source = (PLUGIN_DIR / source_name).read_text(encoding="utf8")
        declarations = re.findall(
            r'^\s*extern\s+"C"\s+bool\s+'
            r"(initFastGpuAsr[A-Za-z0-9_]*)"
            r"\(\)\s+noexcept\s*$",
            source,
            flags=re.MULTILINE,
        )
        assert declarations == [initializers[library_name]], source_name
        namespace_includes = re.findall(
            r'^\s*#include\s+"plugin_namespace\.h"\s*$',
            source,
            flags=re.MULTILINE,
        )
        declared_names = re.findall(
            r"^\s*constexpr\s+char\s+const\*\s+"
            r'k[A-Za-z0-9_]*Name\s*=\s*"([^"]+)";\s*$',
            source,
            flags=re.MULTILINE,
        )
        declared_versions = re.findall(
            r'^\s*constexpr\s+char\s+const\*\s+kPluginVersion\s*=\s*"([^"]+)";\s*$',
            source,
            flags=re.MULTILINE,
        )

        assert len(namespace_includes) == 1, source_name
        assert sorted(declared_names) == sorted(plugin_names), source_name
        assert declared_versions == ["1"], source_name


def test_plugin_dependencies() -> None:
    dependencies = {
        dependency
        for _, plugin_dependencies in PLUGIN_BUILDS
        for dependency in plugin_dependencies
    }

    assert set(CUDA_BUILD_LIBRARIES) == dependencies
    assert len(CUDA_BUILD_LIBRARIES) == len(dependencies)
    assert set(CUDA_BUILD_LIBRARIES) <= set(CUDA_RUNTIME_LIBRARIES)
    assert CUDA_RUNTIME_LIBRARIES == ("cudart", "cublasLt", "cublas", "cufft")
    assert dict(PLUGIN_BUILDS) == {
        "zipformer_attention_value_plugin.cu": ("cublas", "cudart"),
        "zipformer_relative_attention_plugin.cu": ("cublas", "cudart"),
        "zipformer_convolution_plugin.cu": ("cudart",),
        "zipformer_feature_plugin.cu": ("cublas", "cufft", "cudart"),
        "zipformer_resampling_plugin.cu": ("cudart",),
        "zipformer_output_assembly_plugin.cu": ("cudart",),
        "parakeet_feature_plugin.cu": ("cublas", "cufft", "cudart"),
        "parakeet_flash_attention_plugin.cu": ("cublas", "cudart"),
        "parakeet_convolution_plugin.cu": ("cudart",),
    }


def test_nvcc_options_include_supported_architectures_and_ptx_fallback() -> None:
    expected_architectures = (80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121)
    assert (
        *(
            f"--generate-code=arch=compute_{architecture},code=sm_{architecture}"
            for architecture in expected_architectures
        ),
        "--generate-code=arch=compute_80,code=compute_80",
    ) == CUDA_ARCHITECTURE_OPTIONS
    assert (
        "--std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        *CUDA_ARCHITECTURE_OPTIONS,
        "-shared",
    ) == NVCC_OPTIONS


def test_cpp_plugin_namespace_matches_python_exact_bytes(tmp_path: Path) -> None:
    header = (PLUGIN_DIR / "plugin_namespace.h").read_text(encoding="utf8")
    declarations = re.findall(
        r'^\s*constexpr\s+char\s+kPluginNamespace\[\]\s*=\s*"([^"]+)";\s*$',
        header,
        flags=re.MULTILINE,
    )
    assert declarations == [TENSORRT_PLUGIN_NAMESPACE]

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the namespace header test.")

    source_path = tmp_path / "namespace.cpp"
    executable_path = tmp_path / "namespace"
    source_path.write_text(
        """
#include <cstdio>
#include <type_traits>

#include "plugin_namespace.h"
#include "plugin_namespace.h"

using fastgpuasr_tensorrt::kPluginNamespace;
static_assert(std::is_array_v<decltype(kPluginNamespace)>);
static_assert(std::is_const_v<std::remove_extent_t<decltype(kPluginNamespace)>>);

int main()
{
    return std::fwrite(kPluginNamespace, 1, sizeof(kPluginNamespace), stdout)
            == sizeof(kPluginNamespace)
        ? 0
        : 1;
}
""",
        encoding="utf8",
    )
    subprocess.run(
        (
            compiler,
            "--std=c++20",
            "-I",
            str(PLUGIN_DIR),
            str(source_path),
            "-o",
            str(executable_path),
        ),
        check=True,
        timeout=60,
    )

    result = subprocess.run(
        executable_path,
        check=True,
        capture_output=True,
        timeout=60,
    )

    assert result.stdout == TENSORRT_PLUGIN_NAMESPACE.encode("ascii") + b"\0"


def test_parakeet_encoder_frame_limit_matches_cpp_plugin() -> None:
    source = (PLUGIN_DIR / "parakeet_flash_attention_plugin.cu").read_text(
        encoding="utf8"
    )
    declarations = re.findall(
        r"^\s*constexpr\s+int32_t\s+kMaximumSequenceLength\s*=\s*(\d+);\s*$",
        source,
        flags=re.MULTILINE,
    )

    assert declarations == [str(PARAKEET_MAX_ENCODER_FRAMES)]
