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
MODEL_ARTIFACT_NAMES = (
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
)
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


def test_transducer_decoder_types_are_a_subset_of_all_decoder_types() -> None:
    assert DECODER_TYPES == (
        "transducer_modified_beam_search",
        "transducer_greedy_search",
        "ctc_greedy_search",
    )
    assert TRANSDUCER_DECODER_TYPES == (
        "transducer_modified_beam_search",
        "transducer_greedy_search",
    )
    assert set(TRANSDUCER_DECODER_TYPES) < set(DECODER_TYPES)
    assert len(DECODER_TYPES) == len(set(DECODER_TYPES))


def test_serialized_model_contract_constants_are_stable() -> None:
    assert (MODEL_TYPE_PARAKEET, MODEL_TYPE_ZIPFORMER) == (
        "parakeet_asr",
        "zipformer_asr",
    )
    assert MODEL_ARTIFACT_NAMES == (
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


def test_model_artifact_names_are_safe_unique_basenames() -> None:
    assert len(MODEL_ARTIFACT_NAMES) == len(set(MODEL_ARTIFACT_NAMES))
    assert len(MODEL_ARTIFACT_NAMES) == len(
        {name.casefold() for name in MODEL_ARTIFACT_NAMES}
    )
    assert all(Path(name).name == name for name in MODEL_ARTIFACT_NAMES)
    assert all(
        not Path(name).is_absolute() and ".." not in Path(name).parts
        for name in MODEL_ARTIFACT_NAMES
    )
    assert Path(PARAKEET_ONNX_FILE).stem == Path(PARAKEET_TENSORRT_FILE).stem
    assert (
        Path(PARAKEET_DECODER_ONNX_FILE).stem
        == Path(PARAKEET_DECODER_TENSORRT_FILE).stem
    )
    assert Path(ZIPFORMER_ONNX_FILE).stem == Path(ZIPFORMER_TENSORRT_FILE).stem
    assert (
        Path(ZIPFORMER_DECODER_ONNX_FILE).stem
        == Path(ZIPFORMER_DECODER_TENSORRT_FILE).stem
    )


def test_shared_scalar_constants_are_stable() -> None:
    assert ONNX_OPSET_VERSION == 20
    assert TENSORRT_PLUGIN_NAMESPACE == "fast_gpu_asr"
    assert INT32_MAX == 2_147_483_647
    assert ZERO_LOG == -20.7233
    zero_probability = torch.exp(torch.tensor(ZERO_LOG, dtype=torch.float32)).item()
    assert zero_probability == pytest.approx(1e-9, rel=1e-5)


def test_runtime_limits_and_tuning_values_satisfy_algorithm_constraints() -> None:
    assert AUDIO_SAMPLES_PER_WORKER > 0
    assert 0 < PARAKEET_MAX_ENCODER_FRAMES <= INT32_MAX
    assert TDT_SEARCH_CHUNK_STEPS > 0
    assert TDT_SEARCH_CHUNK_STEPS % 2 == 0
    assert 0 < ZIPFORMER_BEAM_SEARCH_THREADS <= 1024
    assert ZIPFORMER_BEAM_SEARCH_THREADS % 32 == 0


def test_plugin_identifiers_are_unique_ascii_abi_keys() -> None:
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

    assert len(plugin_names) == len(set(plugin_names))
    assert all(name.isascii() and name.isidentifier() for name in plugin_names)
    assert TENSORRT_PLUGIN_NAMESPACE.isascii()
    assert TENSORRT_PLUGIN_NAMESPACE.isidentifier()
    assert 0 < len(TENSORRT_PLUGIN_NAMESPACE.encode("ascii")) <= 1024
    assert "\0" not in TENSORRT_PLUGIN_NAMESPACE


def test_every_plugin_build_has_one_runtime_initializer() -> None:
    built_libraries = tuple(
        Path(source_name).with_suffix(".so").name for source_name, _ in PLUGIN_BUILDS
    )
    initialized_libraries = tuple(
        library_name for library_name, _ in PLUGIN_INITIALIZERS
    )

    assert built_libraries == initialized_libraries
    assert len(set(built_libraries)) == len(PLUGIN_BUILDS) == len(PLUGIN_INITIALIZERS)
    assert all(
        initializer.startswith("initFastGpuAsr")
        for _, initializer in PLUGIN_INITIALIZERS
    )
    assert len({initializer for _, initializer in PLUGIN_INITIALIZERS}) == len(
        PLUGIN_INITIALIZERS
    )
    assert all(
        initializer.isascii() and initializer.isidentifier()
        for _, initializer in PLUGIN_INITIALIZERS
    )


def test_plugin_manifest_matches_native_sources_and_exported_initializers() -> None:
    """Keep build metadata synchronized with every native plugin source."""

    source_names = {path.name for path in PLUGIN_DIR.glob("*.cu")}
    manifest_sources = {source_name for source_name, _ in PLUGIN_BUILDS}
    initializers = dict(PLUGIN_INITIALIZERS)

    assert manifest_sources == source_names
    for source_name in sorted(source_names):
        library_name = Path(source_name).with_suffix(".so").name
        source = (PLUGIN_DIR / source_name).read_text(encoding="utf8")
        initializer = initializers[library_name]
        declarations = re.findall(
            rf'^\s*extern\s+"C"\s+bool\s+({re.escape(initializer)})'
            r"\(\)\s+noexcept\s*$",
            source,
            flags=re.MULTILINE,
        )
        assert declarations == [initializer]


def test_plugin_dependencies_are_discoverable_build_libraries() -> None:
    dependencies = {
        dependency
        for _, plugin_dependencies in PLUGIN_BUILDS
        for dependency in plugin_dependencies
    }

    assert set(CUDA_BUILD_LIBRARIES) == dependencies
    assert len(CUDA_BUILD_LIBRARIES) == len(dependencies)


def test_plugin_build_dependency_manifest_is_exact() -> None:
    assert PLUGIN_BUILDS == (
        ("zipformer_attention_value_plugin.cu", ("cublas", "cudart")),
        ("zipformer_relative_attention_plugin.cu", ("cublas", "cudart")),
        ("zipformer_convolution_plugin.cu", ("cudart",)),
        ("zipformer_feature_plugin.cu", ("cublas", "cufft", "cudart")),
        ("zipformer_resampling_plugin.cu", ("cudart",)),
        ("zipformer_output_assembly_plugin.cu", ("cudart",)),
        ("parakeet_feature_plugin.cu", ("cublas", "cufft", "cudart")),
        ("parakeet_flash_attention_plugin.cu", ("cublas", "cudart")),
        ("parakeet_convolution_plugin.cu", ("cudart",)),
    )
    assert all(
        Path(source_name).name == source_name for source_name, _ in PLUGIN_BUILDS
    )
    assert all(
        len(dependencies) == len(set(dependencies)) for _, dependencies in PLUGIN_BUILDS
    )


def test_cuda_architectures_are_unique_and_retain_ptx_fallback() -> None:
    expected_architectures = (80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121)
    assert (
        *(
            f"--generate-code=arch=compute_{architecture},code=sm_{architecture}"
            for architecture in expected_architectures
        ),
        "--generate-code=arch=compute_80,code=compute_80",
    ) == CUDA_ARCHITECTURE_OPTIONS
    assert len(CUDA_ARCHITECTURE_OPTIONS) == len(set(CUDA_ARCHITECTURE_OPTIONS))
    assert CUDA_ARCHITECTURE_OPTIONS[-1] == (
        "--generate-code=arch=compute_80,code=compute_80"
    )
    assert all(
        option.startswith("--generate-code=") for option in CUDA_ARCHITECTURE_OPTIONS
    )
    architecture_start = NVCC_OPTIONS.index(CUDA_ARCHITECTURE_OPTIONS[0])
    assert (
        NVCC_OPTIONS[
            architecture_start : architecture_start + len(CUDA_ARCHITECTURE_OPTIONS)
        ]
        == CUDA_ARCHITECTURE_OPTIONS
    )
    assert (
        "--std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        *CUDA_ARCHITECTURE_OPTIONS,
        "-shared",
    ) == NVCC_OPTIONS


def test_cuda_runtime_dependency_order_is_stable() -> None:
    assert CUDA_RUNTIME_LIBRARIES == ("cudart", "cublasLt", "cublas", "cufft")


def test_cpp_plugin_namespace_matches_python_exact_bytes(tmp_path: Path) -> None:
    """Compile the shared header and verify its exact C++ string contract."""

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
    )

    result = subprocess.run(executable_path, check=True, capture_output=True)

    assert result.stdout == TENSORRT_PLUGIN_NAMESPACE.encode("ascii") + b"\0"


def test_python_plugin_names_match_cpp_creator_names() -> None:
    manifest_sources = {source_name for source_name, _ in PLUGIN_BUILDS}
    assert set(PLUGIN_NAMES_BY_SOURCE) == manifest_sources

    for source_name, plugin_names in PLUGIN_NAMES_BY_SOURCE.items():
        source = (PLUGIN_DIR / source_name).read_text(encoding="utf8")
        declared_names = re.findall(
            r"^\s*constexpr\s+char\s+const\*\s+"
            r'k(?:Plugin|Downsample|Upsample)Name\s*=\s*"([^"]+)";\s*$',
            source,
            flags=re.MULTILINE,
        )
        declared_versions = re.findall(
            r'^\s*constexpr\s+char\s+const\*\s+kPluginVersion\s*=\s*"([^"]+)";\s*$',
            source,
            flags=re.MULTILINE,
        )

        assert tuple(declared_names) == plugin_names
        assert declared_versions == ["1"]


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
