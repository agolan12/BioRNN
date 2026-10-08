#!/bin/bash
# Vanilla vs. plastic RNN on the frequency-encoded Adding Problem.
# One array task = one (model, hidden size, seq_len, seed) run; see
# scripts/plastic_vs_vanilla.py. Submit from the repo root:
#
#   sbatch scripts/plastic_vs_vanilla.slurm
#
# That trains the whole default grid (342 runs) and queues the comparison
# (results.csv + figures) to start once every run has ended. Re-submitting
# after preemptions/failures only trains the missing runs.
#
# Extra arguments go to plastic_vs_vanilla.py. If they change the grid, size
# the array to match (`python scripts/plastic_vs_vanilla.py count ...`):
#   sbatch --array=0-1025%64 scripts/plastic_vs_vanilla.slurm --seeds 0 1 2 --out-dir results/pvv_3seeds
# To redo only the comparison:
#   sbatch --array=0 scripts/plastic_vs_vanilla.slurm compare
#SBATCH --job-name=plastic_vs_vanilla
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --array=0-341%64
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
CONDA_ENV=${CONDA_ENV:-/gscratch/cheme/agolan/phys_417_env}
command=run
if [[ "${1:-}" == compare ]]; then
    command=compare
    shift
fi
echo "job ${SLURM_JOB_ID} ${command} task ${SLURM_ARRAY_TASK_ID:-} on $(hostname)"

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

# The first task queues the comparison behind the whole array (once: not again
# when it is requeued after a preemption).
if [[ $command == run && ${SLURM_ARRAY_TASK_ID:-} == "${SLURM_ARRAY_TASK_MIN:-}" && ${SLURM_RESTART_COUNT:-0} == 0 ]]; then
    sbatch --array=0 --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" --job-name=pvv_compare \
        scripts/plastic_vs_vanilla.slurm compare "$@" \
        || echo "warning: could not queue the comparison; when the runs finish: sbatch --array=0 scripts/plastic_vs_vanilla.slurm compare" >&2
fi

python scripts/plastic_vs_vanilla.py "$command" "$@"
