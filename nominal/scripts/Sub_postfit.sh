#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PROJECT/config/gvv_env.sh"
DATA=${GVV_DATA_DIR:-$PROJECT/../RootSet}
TRUTH_MC=${1:-$DATA/truth_mc.root}
NORMALIZATION_MC=${2:-$DATA/normalization_mc.root}

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [truth_mc.root [normalization_mc.root]]" >&2
    exit 2
fi

cd "$PROJECT" || exit 3
mkdir -p "$PROJECT/results" "$PROJECT/runlog"
export GVV_PROJECT_ROOT="$PROJECT"

sbatch \
    --chdir="$PROJECT" \
    --export="ALL,GVV_PROJECT_ROOT=$PROJECT" \
    "$PROJECT/scripts/subpostgpu.sh" \
    results/fit_result-initial.txt \
    results/Cova_matrix.dat \
    "$TRUTH_MC" \
    "$NORMALIZATION_MC" \
    results/postfit_result
