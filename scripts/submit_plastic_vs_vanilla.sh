#!/bin/bash
# Submit the vanilla-vs-plastic sweep, then the comparison once every run has
# finished. Run from the repo root on the cluster login node. Extra arguments
# are passed to scripts/plastic_vs_vanilla.py, e.g.
#   scripts/submit_plastic_vs_vanilla.sh --seeds 0 1 2 --out-dir results/pvv_3seeds
# Re-running it after preemptions/failures only trains the missing runs.
set -euo pipefail
CONDA_ENV=${CONDA_ENV:-/gscratch/cheme/agolan/phys_417_env}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}

# /etc/bashrc, ~/.bashrc and conda's activate scripts aren't written for
# errexit/nounset: under `set -e` any failing test inside them ends this
# script before it prints or submits anything.
set +eu
source ~/.bashrc
conda activate "$CONDA_ENV" || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }
set -eu
trap 'echo "error: line $LINENO exited with status $?" >&2' ERR  # never die without a message
cd "$(dirname "$0")/.."                        # repo root: sbatch resolves logs/ and scripts/ from here
mkdir -p logs                                  # sbatch won't create it, and a job whose log dir is missing fails with no output

n=$(python scripts/plastic_vs_vanilla.py count "$@")
train=$(sbatch --parsable --array=0-$((n - 1))%${MAX_CONCURRENT} scripts/plastic_vs_vanilla.slurm "$@")
compare=$(sbatch --parsable --dependency=afterany:${train} --job-name=pvv_compare \
    --output=logs/%x_%j.out --error=logs/%x_%j.err scripts/plastic_vs_vanilla.slurm "$@")
echo "training: job ${train} (${n} runs); comparison: job ${compare}"
