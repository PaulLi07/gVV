#!/bin/bash

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
# Slurm executes a spooled copy of this script. The submit side exports the
# canonical repository path so the worker never mistakes the spool for the
# project root.
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

load_fields()
{
    mapfile -t FIT_FIELDS < <(read_fit_fields "$1")
    if [[ ${#FIT_FIELDS[@]} -lt 6 ]]; then
        echo "[GVV] Cannot read the required fields from $1" >&2
        exit 2
    fi
    OUTPUT_TAG=${FIT_FIELDS[0]}
    RESULT_DIR=$(resolve_project_path "${FIT_FIELDS[1]}")
    LOG_DIR=$(resolve_project_path "${FIT_FIELDS[2]}")
    MODEL_FILE=$(resolve_project_path "${FIT_FIELDS[3]}")
    DATA_FILE=$(resolve_project_path "${FIT_FIELDS[4]}")
    NORMALIZATION_FILE=$(resolve_project_path "${FIT_FIELDS[5]}")
    BACKGROUND_FILES=()
    for ((index = 6; index < ${#FIT_FIELDS[@]}; ++index)); do
        BACKGROUND_FILES+=("$(resolve_project_path "${FIT_FIELDS[index]}")")
    done
    RESULT_FILE="$RESULT_DIR/fit_result-$OUTPUT_TAG.txt"
    COVARIANCE_FILE="$RESULT_DIR/Cova_matrix-$OUTPUT_TAG.dat"
    PROJECTION_FILE="$RESULT_DIR/projection-$OUTPUT_TAG.root"
    LOG_FILE="$LOG_DIR/fit-$OUTPUT_TAG.log"
}

validate_inputs()
{
    local input_file
    for input_file in \
        "$1" "$MODEL_FILE" "$DATA_FILE" "$NORMALIZATION_FILE" \
        "${BACKGROUND_FILES[@]}"
    do
        if [[ ! -r $input_file ]]; then
            echo "[GVV] Input is missing or unreadable: $input_file" >&2
            exit 4
        fi
    done
}

run_worker()
{
    local fit_config=$1
    load_fields "$fit_config"
    validate_inputs "$fit_config"
    mkdir -p -- "$RESULT_DIR" "$LOG_DIR"
    if [[ ! -w $RESULT_DIR || ! -w $LOG_DIR ]]; then
        echo "[GVV] Output directories are not writable" >&2
        exit 6
    fi
    if [[ ! -x $PROJECT_DIR/bin/Fit.exe ]]; then
        echo "[GVV] Build bin/Fit.exe before submission" >&2
        exit 8
    fi
    source "$PROJECT_DIR/config/gvv_env.sh"
    cd "$PROJECT_DIR"

    echo "[GVV] Start: $(date --iso-8601=seconds)"
    echo "[GVV] Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
    echo "[GVV] Host: $(hostname -f)"
    echo "[GVV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "[GVV] CUDA root: $GVV_CUDA_ROOT"
    echo "[GVV] ROOT: $GVV_ROOTSYS"
    echo "[GVV] Fit configuration: $fit_config"
    echo "[GVV] Output tag: $OUTPUT_TAG"
    /usr/bin/nvidia-smi

    printf '[GVV] Command:'
    printf ' %q' "$PROJECT_DIR/bin/Fit.exe" "$fit_config"
    printf '\n'
    srun --ntasks=1 "$PROJECT_DIR/bin/Fit.exe" "$fit_config"

    local output_file
    for output_file in \
        "$RESULT_FILE" "$COVARIANCE_FILE" "$PROJECTION_FILE"
    do
        if [[ ! -s $output_file ]]; then
            echo "[GVV] Fit returned success but output is missing: $output_file" >&2
            exit 10
        fi
    done
    echo "[GVV] End: $(date --iso-8601=seconds)"
    echo "[GVV] Fit result: $RESULT_FILE"
    echo "[GVV] Covariance matrix: $COVARIANCE_FILE"
    echo "[GVV] Projection: $PROJECTION_FILE"
}

if [[ ${1:-} == "--worker" ]]; then
    if [[ $# -ne 2 ]]; then
        usage
        exit 2
    fi
    run_worker "$(readlink -f -- "$2")"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi
FIT_CONFIG=$(readlink -f -- "$(resolve_project_path "${1:-config/fit.json}")")
load_fields "$FIT_CONFIG"
validate_inputs "$FIT_CONFIG"
mkdir -p -- "$RESULT_DIR" "$LOG_DIR"
if [[ ! -x $PROJECT_DIR/bin/Fit.exe ]]; then
    echo "[GVV] Build bin/Fit.exe before submission" >&2
    exit 8
fi

JOB_ID=$(sbatch --parsable \
    --chdir="$PROJECT_DIR" \
    --export="ALL,GVV_PROJECT_ROOT=$PROJECT_DIR" \
    --output="$LOG_FILE" \
    --open-mode=truncate \
    "$SCRIPT_PATH" --worker "$FIT_CONFIG")
echo "[GVV] Submitted Slurm job $JOB_ID"
echo "[GVV] Log: $LOG_FILE"
