#!/bin/bash

#SBATCH --partition=gpupwa
#SBATCH --qos=pwadedicate
#SBATCH --account=gpupwa
#SBATCH --job-name=gvv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=runlog/gpujob-%j.out
#SBATCH --mem-per-cpu=24288
#SBATCH --gres=gpu:a100:1

set -u -o pipefail

PROJECT_DIR=${GVV_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -r "$PROJECT_DIR/config/gvv_env.sh" ]]; then
    echo "[GVV] Cannot resolve the project root: $PROJECT_DIR" >&2
    echo "[GVV] Submit through scripts/Sub.sh or export GVV_PROJECT_ROOT." >&2
    exit 3
fi
if ! source "$PROJECT_DIR/config/gvv_env.sh"; then
    echo "[GVV] Failed to load $PROJECT_DIR/config/gvv_env.sh" >&2
    exit 3
fi

usage()
{
    echo "Usage: sbatch $0 data.root normalization_mc.root SB1.root SB2.root [fit_result.txt [n_starts [base_seed]]]" >&2
}

if [[ $# -lt 4 || $# -gt 7 ]]; then
    usage
    exit 2
fi

submission_dir=${SLURM_SUBMIT_DIR:-$PWD}

resolve_from_submission_dir()
{
    if [[ $1 = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$submission_dir" "$1"
    fi
}

data_file=$(resolve_from_submission_dir "$1")
normalization_mc_file=$(resolve_from_submission_dir "$2")
sb1_file=$(resolve_from_submission_dir "$3")
sb2_file=$(resolve_from_submission_dir "$4")

if [[ $# -ge 5 ]]; then
    result_file=$(resolve_from_submission_dir "$5")
else
    result_file="$PROJECT_DIR/results/fit_result-${SLURM_JOB_ID:-manual}.txt"
fi

n_starts=${6:-1}
base_seed=${7:-20260815}
if [[ ! $n_starts =~ ^[1-9][0-9]*$ ]]; then
    echo "[GVV] n_starts must be a positive integer: $n_starts" >&2
    exit 2
fi
if [[ ! $base_seed =~ ^[0-9]+$ ]]; then
    echo "[GVV] base_seed must be a non-negative integer: $base_seed" >&2
    exit 2
fi

if [[ ! -d $PROJECT_DIR ]]; then
    echo "[GVV] Project directory is missing: $PROJECT_DIR" >&2
    exit 3
fi

for input_file in \
    "$data_file" \
    "$normalization_mc_file" \
    "$sb1_file" \
    "$sb2_file"
do
    if [[ ! -r $input_file ]]; then
        echo "[GVV] Input file is missing or unreadable: $input_file" >&2
        exit 4
    fi
done

result_dir=$(dirname -- "$result_file")
if ! mkdir -p -- "$result_dir" || [[ ! -w $result_dir ]]; then
    echo "[GVV] Result directory is not writable: $result_dir" >&2
    exit 6
fi

cd "$PROJECT_DIR" || exit 7
if [[ ! -x "$PROJECT_DIR/bin/Fit.exe" ]]; then
    echo "[GVV] bin/Fit.exe is missing or not executable in $PROJECT_DIR" >&2
    exit 8
fi

echo "[GVV] Start: $(date --iso-8601=seconds)"
echo "[GVV] Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "[GVV] Host: $(hostname -f)"
echo "[GVV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[GVV] CUDA root: $GVV_CUDA_ROOT"
echo "[GVV] ROOT: $GVV_ROOTSYS"
echo "[GVV] Multistart fits: $n_starts"
echo "[GVV] Base seed: $base_seed"

if ! /usr/bin/nvidia-smi; then
    echo "[GVV] nvidia-smi failed; GPU allocation is not usable" >&2
    exit 9
fi

printf '[GVV] Command:'
printf ' %q' \
    "$PROJECT_DIR/bin/Fit.exe" \
    "$data_file" \
    "$normalization_mc_file" \
    "$sb1_file" \
    "$sb2_file" \
    "$result_file" \
    "$n_starts" \
    "$base_seed"
printf '\n'

srun --ntasks=1 "$PROJECT_DIR/bin/Fit.exe" \
    "$data_file" \
    "$normalization_mc_file" \
    "$sb1_file" \
    "$sb2_file" \
    "$result_file" \
    "$n_starts" \
    "$base_seed"
fit_status=$?

if [[ $fit_status -ne 0 ]]; then
    echo "[GVV] Fit.exe failed with exit code $fit_status" >&2
    exit "$fit_status"
fi

if [[ ! -s $result_file ]]; then
    echo "[GVV] Fit.exe returned success but the result file is missing or empty" >&2
    exit 10
fi

if [[ ! -s results/Cova_matrix.dat ]]; then
    echo "[GVV] Fit.exe returned success but results/Cova_matrix.dat is missing or empty" >&2
    exit 11
fi

if [[ ! -s results/projection0.root ]]; then
    echo "[GVV] Fit.exe returned success but results/projection0.root is missing or empty" >&2
    exit 12
fi

echo "[GVV] End: $(date --iso-8601=seconds)"
echo "[GVV] Fit result: $result_file"
echo "[GVV] Covariance matrix: $PROJECT_DIR/results/Cova_matrix.dat"
echo "[GVV] Projection: $PROJECT_DIR/results/projection0.root"
