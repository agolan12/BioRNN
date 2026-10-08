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
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-compare} on $(hostname)"

# /etc/bashrc, ~/.bashrc and conda's activate scripts aren't written for
# errexit/nounset: under `set -e` any failing test inside them ends the job
# with empty logs.
set +eu
source ~/.bashrc
conda activate "$CONDA_ENV" || { echo "error: could not activate $CONDA_ENV" >&2; exit 1; }
set -eu
trap 'echo "error: line $LINENO exited with status $?" >&2' ERR  # never die without a message
cd "$SLURM_SUBMIT_DIR"                         # repo root
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"  # import biornn without pip install -e

echo "python: $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python scripts/plastic_vs_vanilla.py run "$@"
else
    python scripts/plastic_vs_vanilla.py compare "$@"
fi
