#!/usr/bin/env bash
# AVO wrapper for the DeepSeek V4 MoE M-grouped contiguous FP4 case.
#
# DGX Spark normally uses the C++ extension built in this checkout and imports
# Python/JIT headers from this checkout so local code edits are reflected in
# benchmark results.
#
# Alternative environments can point VENV_PY at a local environment and either
# use a built _C extension in this checkout or set DSV4_DEEP_GEMM to an
# external checkout containing deep_gemm/_C*.so.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSV4_DEEP_GEMM="${DSV4_DEEP_GEMM:-${REPO_ROOT}}"
DSV4_PYDEPS="${DSV4_PYDEPS:-${REPO_ROOT}}"
VENV_PY="${VENV_PY:-${PYTHON:-python3}}"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="${PYTHON:-python3}"
fi

shopt -s nullglob
local_ext=("${REPO_ROOT}"/deep_gemm/_C*.so)
if (( ${#local_ext[@]} == 0 )); then
  prebuilt_ext=("${DSV4_DEEP_GEMM}"/deep_gemm/_C*.so)
  if (( ${#prebuilt_ext[@]} == 0 )); then
    echo "Could not find deep_gemm extension under ${REPO_ROOT}/deep_gemm or ${DSV4_DEEP_GEMM}/deep_gemm" >&2
    exit 1
  fi
  ln -sf "${prebuilt_ext[0]}" "${REPO_ROOT}/deep_gemm/$(basename "${prebuilt_ext[0]}")"
fi
mkdir -p "${REPO_ROOT}/third-party/cutlass" "${REPO_ROOT}/third-party/fmt"
if [[ ! -e "${REPO_ROOT}/third-party/cutlass/include" && -e "${DSV4_DEEP_GEMM}/third-party/cutlass/include" ]]; then
  ln -s "${DSV4_DEEP_GEMM}/third-party/cutlass/include" "${REPO_ROOT}/third-party/cutlass/include"
fi
if [[ ! -e "${REPO_ROOT}/third-party/fmt/include" && -e "${DSV4_DEEP_GEMM}/third-party/fmt/include" ]]; then
  ln -s "${DSV4_DEEP_GEMM}/third-party/fmt/include" "${REPO_ROOT}/third-party/fmt/include"
fi
for cutlass_dir in cute cutlass; do
  if [[ ! -e "${REPO_ROOT}/deep_gemm/include/${cutlass_dir}" && -e "${DSV4_DEEP_GEMM}/deep_gemm/include/${cutlass_dir}" ]]; then
    ln -s "${DSV4_DEEP_GEMM}/deep_gemm/include/${cutlass_dir}" "${REPO_ROOT}/deep_gemm/include/${cutlass_dir}"
  fi
done

if [[ -d "$DSV4_PYDEPS" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${DSV4_PYDEPS}:${REPO_ROOT}/tests:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tests:${PYTHONPATH:-}"
fi
export DG_JIT_CACHE_DIR="${REPO_ROOT}/avo/.dg_cache_mgroup"
mkdir -p "$DG_JIT_CACHE_DIR"

stage="${1:-preflight}"
iter="${2:-0}"
variant="${3:-baseline_mgroup_moe}"

exec "$VENV_PY" "$REPO_ROOT/avo/bench_mgroup_moe.py" "$stage" --iter "$iter" --variant "$variant"
