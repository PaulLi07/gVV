#!/bin/bash

# Usage: bash submit_fit.sh config/fit.json
# USER SETTINGS: scheduler resources are grouped in the SBATCH block below.
# Physics and minimizer settings belong in JSON; no source edit is needed.

# Slurm submission and worker entry point for amplitude fitting only.
# Post Calculation is submitted independently through submit_post.sh.
#SBATCH --partition=gpupwa
#SBATCH --qos=pwadedicate
#SBATCH --account=gpupwa
#SBATCH --job-name=gvv-fit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24288
#SBATCH --gres=gpu:a100:1

set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT_DIR=${GVV_PROJECT_ROOT:-$SCRIPT_DIR}

usage()
{
    echo "Usage: $0 [config/fit.json]" >&2
}

resolve_project_path()
{
    if [[ $1 = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$PROJECT_DIR" "$1"
    fi
}

read_fit_fields()
{
    python3 - "$1" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
output = config["output"]
tag = output["tag"]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
    raise SystemExit("unsafe output.tag in fit configuration")
print(tag)
print(output["directory"])
print(output["log_directory"])
print(config["model"])
inputs = config["inputs"]
print(inputs["data"])
print(inputs["normalization_mc"])
for sample in inputs.get("backgrounds", []):
    print(sample["file"])
PY
}

load_fit_fields()
{
    mapfile -t FIT_FIELDS < <(read_fit_fields "$1")
    FIT_TAG=${FIT_FIELDS[0]}
    FIT_RESULT_DIR=$(resolve_project_path "${FIT_FIELDS[1]}")
    FIT_LOG_DIR=$(resolve_project_path "${FIT_FIELDS[2]}")
    FIT_MODEL=$(resolve_project_path "${FIT_FIELDS[3]}")
    FIT_INPUTS=(
        "$(resolve_project_path "${FIT_FIELDS[4]}")"
        "$(resolve_project_path "${FIT_FIELDS[5]}")")
    for ((index = 6; index < ${#FIT_FIELDS[@]}; ++index)); do
        FIT_INPUTS+=("$(resolve_project_path "${FIT_FIELDS[index]}")")
    done
    FIT_REPORT="$FIT_RESULT_DIR/fit_result-$FIT_TAG.txt"
    FIT_STATE="$FIT_RESULT_DIR/fit_state-$FIT_TAG.json"
    FIT_PROJECTION="$FIT_RESULT_DIR/projection-$FIT_TAG.root"
    FIT_LOG="$FIT_LOG_DIR/fit-$FIT_TAG.log"
}

require_files()
{
    local file
    for file in "$@"; do
        if [[ ! -r $file ]]; then
            echo "[GVV] Input is missing or unreadable: $file" >&2
            exit 4
        fi
    done
}

load_environment()
{
    source "$PROJECT_DIR/config/gvv_env.sh"
    cd "$PROJECT_DIR"
    echo "[GVV] Start: $(date --iso-8601=seconds)"
    echo "[GVV] Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
    echo "[GVV] Host: $(hostname -f)"
    echo "[GVV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    /usr/bin/nvidia-smi
}

run_worker()
{
    local config=$1
    load_fit_fields "$config"
    require_files "$config" "$FIT_MODEL" "${FIT_INPUTS[@]}"
    mkdir -p -- "$FIT_RESULT_DIR" "$FIT_LOG_DIR"
    [[ -x $PROJECT_DIR/bin/Fit.exe ]] || {
        echo "[GVV] Build bin/Fit.exe with make before submission" >&2
        exit 8
    }

    load_environment
    srun --ntasks=1 "$PROJECT_DIR/bin/Fit.exe" "$config"

    local product
    for product in "$FIT_REPORT" "$FIT_STATE" "$FIT_PROJECTION"; do
        [[ -s $product ]] || {
            echo "[GVV] Fit output is missing: $product" >&2
            exit 10
        }
    done
    echo "[GVV] Fit report: $FIT_REPORT"
    echo "[GVV] Fit state: $FIT_STATE"
    echo "[GVV] Projection: $FIT_PROJECTION"
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 2 ]] || { usage; exit 2; }
    run_worker "$(readlink -f -- "$2")"
    exit 0
fi

[[ $# -le 1 ]] || { usage; exit 2; }
CONFIG=$(readlink -f -- "$(resolve_project_path "${1:-config/fit.json}")")
load_fit_fields "$CONFIG"
require_files "$CONFIG" "$FIT_MODEL" "${FIT_INPUTS[@]}"
mkdir -p -- "$FIT_RESULT_DIR" "$FIT_LOG_DIR"
[[ -x $PROJECT_DIR/bin/Fit.exe ]] || {
    echo "[GVV] Build bin/Fit.exe with make before submission" >&2
    exit 8
}

JOB_ID=$(sbatch --parsable \
    --chdir="$PROJECT_DIR" \
    --export="ALL,GVV_PROJECT_ROOT=$PROJECT_DIR" \
    --output="$FIT_LOG" --open-mode=truncate \
    "$SCRIPT_PATH" --worker "$CONFIG")
echo "[GVV] Submitted fit job $JOB_ID"
echo "[GVV] Log: $FIT_LOG"
