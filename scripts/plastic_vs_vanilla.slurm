#!/bin/bash
# Vanilla vs. plastic RNN on the frequency-encoded Adding Problem.
# One array task = one (model, hidden size, seq_len, seed) run; see
# scripts/plastic_vs_vanilla.py. Submit with scripts/submit_plastic_vs_vanilla.sh,
# which sizes the array and queues the comparison job after it. Without an
# array task id (the comparison job) this script runs `compare` instead.
#SBATCH --job-name=plastic_vs_vanilla
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
CONDA_ENV=${CONDA_ENV:-/gscratch/cheme/agolan/phys_417_env}

source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"                         # repo root
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"  # import biornn without pip install -e

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-compare} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python scripts/plastic_vs_vanilla.py run "$@"
else
    python scripts/plastic_vs_vanilla.py compare "$@"
fi
