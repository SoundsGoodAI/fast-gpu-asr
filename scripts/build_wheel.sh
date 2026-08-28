#!/usr/bin/env bash
# Copyright SoundsGoodAI 2026 - Daniil Kulko

set -euo pipefail

if [[ "$#" -gt 1 ]]; then
    printf 'Usage: %s [wheel-directory]\n' "${0##*/}" >&2
    exit 2
fi

repository_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -eq 1 && "$1" != /* ]]; then
    wheel_dir="${PWD}/$1"
else
    wheel_dir="${1:-${repository_dir}/dist}"
fi
mkdir -p "${wheel_dir}"
build_dir="$(mktemp -d "${wheel_dir}/.build-wheel.XXXXXX")"
raw_wheel_dir="${build_dir}/raw"
repaired_wheel_dir="${build_dir}/repaired"
source_dir="${build_dir}/source"
mkdir "${raw_wheel_dir}" "${repaired_wheel_dir}" "${source_dir}"
trap 'rm -rf "${build_dir}"' EXIT

cd "${repository_dir}"
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
cp -a LICENSE README.md pyproject.toml setup.py src "${source_dir}/"

cd "${source_dir}"
uv build --wheel --out-dir "${raw_wheel_dir}"

mapfile -t raw_wheels < <(
    find "${raw_wheel_dir}" -maxdepth 1 -type f -name '*.whl' -print
)
if [[ "${#raw_wheels[@]}" -ne 1 ]]; then
    printf 'Expected exactly one raw wheel, found %s.\n' "${#raw_wheels[@]}" >&2
    exit 1
fi

cd "${repository_dir}"
uvx --from auditwheel auditwheel repair \
    --plat manylinux_2_27_x86_64 \
    --wheel-dir "${repaired_wheel_dir}" \
    --exclude libnvinfer.so.11 \
    --exclude libcudart.so.13 \
    --exclude libcublas.so.13 \
    --exclude libcublasLt.so.13 \
    --exclude libcufft.so.12 \
    "${raw_wheels[0]}"

mapfile -t repaired_wheels < <(
    find "${repaired_wheel_dir}" -maxdepth 1 -type f -name '*.whl' -print
)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
    printf 'Expected exactly one repaired wheel, found %s.\n' \
        "${#repaired_wheels[@]}" >&2
    exit 1
fi

published_wheel="${wheel_dir}/${repaired_wheels[0]##*/}"
mv -- "${repaired_wheels[0]}" "${published_wheel}"
find "${wheel_dir}" -maxdepth 1 -type f -name 'fast_gpu_asr-*.whl' \
    ! -name "${published_wheel##*/}" -delete
