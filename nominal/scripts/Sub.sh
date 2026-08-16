#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PROJECT/config/gvv_env.sh"
DATA=${GVV_DATA_DIR:-$PROJECT/../RootSet}
N_STARTS=${1:-10}
BASE_SEED=${2:-20260815}

if [[ $# -gt 2 || ! $N_STARTS =~ ^[1-9][0-9]*$ || ! $BASE_SEED =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [n_starts=10] [base_seed=20260815]" >&2
    exit 2
fi

cd "$PROJECT" || exit 3
mkdir -p "$PROJECT/results" "$PROJECT/runlog"
export GVV_PROJECT_ROOT="$PROJECT"

sbatch \
    --chdir="$PROJECT" \
    --export="ALL,GVV_PROJECT_ROOT=$PROJECT" \
    "$PROJECT/scripts/subgpu.sh" \
    "$DATA/data.root" \
    "$DATA/normalization_mc.root" \
    "$DATA/SB1.root" \
    "$DATA/SB2.root" \
    "$PROJECT/results/fit_result-initial.txt" \
    "$N_STARTS" \
    "$BASE_SEED"
