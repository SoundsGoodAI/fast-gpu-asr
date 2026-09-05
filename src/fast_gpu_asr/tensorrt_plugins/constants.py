#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Constants shared by TensorRT plugin building and loading."""

CUDA_ARCHITECTURE_OPTIONS = (
    "--generate-code=arch=compute_75,code=sm_75",
    "--generate-code=arch=compute_80,code=sm_80",
    "--generate-code=arch=compute_86,code=sm_86",
    "--generate-code=arch=compute_87,code=sm_87",
    "--generate-code=arch=compute_88,code=sm_88",
    "--generate-code=arch=compute_89,code=sm_89",
    "--generate-code=arch=compute_90,code=sm_90",
    "--generate-code=arch=compute_100,code=sm_100",
    "--generate-code=arch=compute_103,code=sm_103",
    "--generate-code=arch=compute_110,code=sm_110",
    "--generate-code=arch=compute_120,code=sm_120",
    "--generate-code=arch=compute_121,code=sm_121",
    "--generate-code=arch=compute_80,code=compute_80",
)
NVCC_OPTIONS = (
    "--std=c++20",
    "-O3",
    "-DNDEBUG",
    "-Xcompiler=-fPIC",
    *CUDA_ARCHITECTURE_OPTIONS,
    "-shared",
)

# Dependencies use cuda-pathfinder's short names rather than linker search flags.
PLUGIN_BUILDS = (
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
CUDA_BUILD_LIBRARIES = ("cudart", "cublas", "cufft")
CUDA_RUNTIME_LIBRARIES = ("cudart", "cublasLt", "cublas", "cufft")

PLUGIN_INITIALIZERS = (
    (
        "zipformer_attention_value_plugin.so",
        "initFastGpuAsrZipformerAttentionValuePlugin",
    ),
    (
        "zipformer_relative_attention_plugin.so",
        "initFastGpuAsrZipformerRelativeAttentionPlugin",
    ),
    ("zipformer_convolution_plugin.so", "initFastGpuAsrZipformerConvolutionPlugin"),
    ("zipformer_feature_plugin.so", "initFastGpuAsrZipformerFeaturePlugin"),
    ("zipformer_resampling_plugin.so", "initFastGpuAsrZipformerResamplingPlugins"),
    (
        "zipformer_output_assembly_plugin.so",
        "initFastGpuAsrZipformerOutputAssemblyPlugin",
    ),
    ("parakeet_feature_plugin.so", "initFastGpuAsrParakeetFeaturePlugin"),
    (
        "parakeet_flash_attention_plugin.so",
        "initFastGpuAsrParakeetFlashAttentionPlugin",
    ),
    ("parakeet_convolution_plugin.so", "initFastGpuAsrParakeetConvolutionPlugin"),
)
