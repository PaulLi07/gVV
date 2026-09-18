#!/bin/bash

# Usage: bash submit_post.sh results/fit_state-TAG.json /path/to/truth.root RootSet/normalization_mc.root
# USER SETTINGS: scheduler resources are grouped in the SBATCH block below.
# Physics and minimizer settings belong in JSON; no source edit is needed.

# Slurm submission and worker entry point for Post Calculation only.
# Projection plotting remains a login-node ROOT task driven by
# post/plotting/draw.sh.
#SBATCH --partition=gpupwa
#SBATCH --qos=pwadedicate
#SBATCH --account=gpupwa
#SBATCH --job-name=gvv-post
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
    echo "Usage: $0 fit_state.json truth_mc.root normalization_mc.root" >&2
}

resolve_project_path()
{
    if [[ $1 = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$PROJECT_DIR" "$1"
    fi
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

read_post_tag()
{
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["output_tag"])
PY
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
    local state=$1 truth=$2 normalization=$3
    require_files "$state" "$truth" "$normalization"
    [[ -x $PROJECT_DIR/bin/Post.exe ]] || {
        echo "[GVV] Build bin/Post.exe with 'make post' before submission" >&2
        exit 8
    }

    load_environment
    srun --ntasks=1 "$PROJECT_DIR/bin/Post.exe" \
        "$state" "$truth" "$normalization"

    local tag output
    tag=$(read_post_tag "$state")
    output="$PROJECT_DIR/post/calculation/results/post_result-$tag"
    [[ -s $output.txt && -s $output.root ]] || {
        echo "[GVV] Post Calculation output is incomplete: $output" >&2
        exit 10
    }
    echo "[GVV] Post Calculation results: $output.{txt,root}"
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 4 ]] || { usage; exit 2; }
    run_worker \
        "$(readlink -f -- "$2")" "$(readlink -f -- "$3")" \
        "$(readlink -f -- "$4")"
    exit 0
fi

[[ $# -eq 3 ]] || { usage; exit 2; }
STATE=$(readlink -f -- "$(resolve_project_path "$1")")
TRUTH=$(readlink -f -- "$(resolve_project_path "$2")")
NORMALIZATION=$(readlink -f -- "$(resolve_project_path "$3")")
require_files "$STATE" "$TRUTH" "$NORMALIZATION"
[[ -x $PROJECT_DIR/bin/Post.exe ]] || {
    echo "[GVV] Build bin/Post.exe with 'make post' before submission" >&2
    exit 8
}

POST_TAG=$(read_post_tag "$STATE")
POST_LOG_DIR="$PROJECT_DIR/runlog"
POST_LOG="$POST_LOG_DIR/post-$POST_TAG.log"
mkdir -p -- "$POST_LOG_DIR" "$PROJECT_DIR/post/calculation/results"

JOB_ID=$(sbatch --parsable \
    --chdir="$PROJECT_DIR" \
    --export="ALL,GVV_PROJECT_ROOT=$PROJECT_DIR" \
    --output="$POST_LOG" --open-mode=truncate \
    "$SCRIPT_PATH" --worker "$STATE" "$TRUTH" "$NORMALIZATION")
echo "[GVV] Submitted Post Calculation job $JOB_ID"
echo "[GVV] Log: $POST_LOG"
