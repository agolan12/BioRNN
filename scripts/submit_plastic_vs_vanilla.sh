#!/bin/bash
# Submit the vanilla-vs-plastic sweep, then the comparison once every run has
# finished. Run from the repo root on the cluster login node. Extra arguments
# are passed to scripts/plastic_vs_vanilla.py, e.g.
#   scripts/submit_plastic_vs_vanilla.sh --seeds 0 1 2 --out-dir results/pvv_3seeds
# Re-running it after preemptions/failures only trains the missing runs.
set -euo pipefail
CONDA_ENV=${CONDA_ENV:-/gscratch/cheme/agolan/phys_417_env}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}

set +u                                         # /etc/bashrc and conda activate scripts reference unset vars
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u
mkdir -p logs

n=$(python scripts/plastic_vs_vanilla.py count "$@")
train=$(sbatch --parsable --array=0-$((n - 1))%${MAX_CONCURRENT} scripts/plastic_vs_vanilla.slurm "$@")
compare=$(sbatch --parsable --dependency=afterany:${train} --job-name=pvv_compare \
    --output=logs/%x_%j.out --error=logs/%x_%j.err scripts/plastic_vs_vanilla.slurm "$@")
echo "training: job ${train} (${n} runs); comparison: job ${compare}"
