#!/bin/bash

#SBATCH --partition=gpupwa
#SBATCH --qos=pwadedicate
#SBATCH --account=gpupwa
#SBATCH --job-name=gvv-postfit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=runlog/postfit-%j.out
#SBATCH --mem-per-cpu=24288
#SBATCH --gres=gpu:a100:1

set -u -o pipefail

PROJECT_DIR=${GVV_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -r "$PROJECT_DIR/config/gvv_env.sh" ]]; then
    echo "[GVV PostFit] Cannot resolve the project root: $PROJECT_DIR" >&2
    echo "[GVV PostFit] Submit through scripts/Sub_postfit.sh or export GVV_PROJECT_ROOT." >&2
    exit 3
fi
if ! source "$PROJECT_DIR/config/gvv_env.sh"; then
    echo "[GVV PostFit] Failed to load $PROJECT_DIR/config/gvv_env.sh" >&2
    exit 3
fi

usage()
{
    echo "Usage: sbatch $0 fit_result.txt Cova_matrix.dat truth_mc.root normalization_mc.root [output_prefix]" >&2
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
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

fit_result=$(resolve_from_submission_dir "$1")
covariance=$(resolve_from_submission_dir "$2")
truth_mc=$(resolve_from_submission_dir "$3")
normalization_mc=$(resolve_from_submission_dir "$4")
output_prefix=$(resolve_from_submission_dir "${5:-results/postfit_result}")

for input_file in "$fit_result" "$covariance" "$truth_mc" "$normalization_mc"
do
    if [[ ! -r $input_file ]]; then
        echo "[GVV PostFit] Input is missing or unreadable: $input_file" >&2
        exit 4
    fi
done

output_dir=$(dirname -- "$output_prefix")
if ! mkdir -p -- "$output_dir" || [[ ! -w $output_dir ]]; then
    echo "[GVV PostFit] Output directory is not writable: $output_dir" >&2
    exit 5
fi

cd "$PROJECT_DIR" || exit 6
if [[ ! -x "$PROJECT_DIR/bin/PostFit.exe" ]]; then
    echo "[GVV PostFit] bin/PostFit.exe is missing or not executable" >&2
    exit 7
fi

echo "[GVV PostFit] Start: $(date --iso-8601=seconds)"
echo "[GVV PostFit] Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "[GVV PostFit] Host: $(hostname -f)"
echo "[GVV PostFit] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[GVV PostFit] CUDA root: $GVV_CUDA_ROOT"
echo "[GVV PostFit] ROOT: $GVV_ROOTSYS"
printf '[GVV PostFit] Command:'
printf ' %q' "$PROJECT_DIR/bin/PostFit.exe" "$fit_result" "$covariance" \
    "$truth_mc" "$normalization_mc" "$output_prefix"
printf '\n'

if ! /usr/bin/nvidia-smi; then
    echo "[GVV PostFit] nvidia-smi failed" >&2
    exit 8
fi

srun --ntasks=1 "$PROJECT_DIR/bin/PostFit.exe" \
    "$fit_result" \
    "$covariance" \
    "$truth_mc" \
    "$normalization_mc" \
    "$output_prefix"
postfit_status=$?
if [[ $postfit_status -ne 0 ]]; then
    echo "[GVV PostFit] PostFit.exe failed with $postfit_status" >&2
    exit "$postfit_status"
fi

for output_file in \
    "${output_prefix}.txt" \
    "${output_prefix}.root" \
    "${output_dir}/fit_fractions.tex"
do
    if [[ ! -s $output_file ]]; then
        echo "[GVV PostFit] Missing output: $output_file" >&2
        exit 9
    fi
done

echo "[GVV PostFit] End: $(date --iso-8601=seconds)"
echo "[GVV PostFit] Output prefix: $output_prefix"
