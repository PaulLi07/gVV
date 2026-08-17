#!/usr/bin/env bash

# Project-local runtime/build environment for gVV.
# This file is intentionally independent of the global setup_ctpwa function.
# It records the paths used by this project without changing the user's shell
# configuration or attempting to resolve the site's CUDA module-version policy.

GVV_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export GVV_PROJECT_ROOT=$(cd -- "$GVV_CONFIG_DIR/.." && pwd)

# Keep the current project compiler/runtime paths explicit.
export GVV_CUDA_ROOT="${GVV_CUDA_ROOT:-/usr/local/cuda-12}"
export GVV_ROOTSYS="${GVV_ROOTSYS:-/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt}"
# Convenience path for interactive inspection. Fit.exe itself takes every
# input path from config/fit.json and does not implicitly read this variable.
export GVV_DATA_DIR="${GVV_DATA_DIR:-$GVV_PROJECT_ROOT/RootSet}"

if [[ ! -x "$GVV_CUDA_ROOT/bin/nvcc" ]]; then
    echo "[GVV env] nvcc is unavailable: $GVV_CUDA_ROOT/bin/nvcc" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ! -r "$GVV_ROOTSYS/bin/thisroot.sh" ]]; then
    echo "[GVV env] ROOT setup script is unavailable: $GVV_ROOTSYS/bin/thisroot.sh" >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$GVV_CUDA_ROOT"
export CUDAROOT="$GVV_CUDA_ROOT"
export ROOTSYS="$GVV_ROOTSYS"
export NVCC="$GVV_CUDA_ROOT/bin/nvcc"
export ROOT_PREFIX="$GVV_ROOTSYS"

source "$GVV_ROOTSYS/bin/thisroot.sh" || return 1 2>/dev/null || exit 1
export PATH="$GVV_ROOTSYS/bin:$GVV_CUDA_ROOT/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib:$GVV_CUDA_ROOT/lib64:$GVV_ROOTSYS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hash -r

if [[ "${GVV_ENV_QUIET:-0}" != 1 ]]; then
    echo "[GVV env] project: $GVV_PROJECT_ROOT"
    echo "[GVV env] CUDA root: $GVV_CUDA_ROOT"
    echo "[GVV env] ROOT: $GVV_ROOTSYS"
    echo "[GVV env] nvcc: $($GVV_CUDA_ROOT/bin/nvcc --version | tail -n 1)"
    echo "[GVV env] ROOT version: $($GVV_ROOTSYS/bin/root-config --version)"
fi
