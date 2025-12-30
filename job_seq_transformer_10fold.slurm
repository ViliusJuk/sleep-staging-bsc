#!/bin/bash
#SBATCH --job-name=seqtr_cv10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/seqtr_cv_%A_%a.out
#SBATCH --error=logs/seqtr_cv_%A_%a.err

set -euo pipefail

cd ~/projects/sleep-staging-bsc
source .venv/bin/activate

mkdir -p logs results models

echo "START: $(date)"
echo "JOB: ${SLURM_JOB_ID} TASK: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

# --- paleidžiam vieną fold ---
python -u -m src.sleepstaging.train_seq_transformer_cv \
  --k 10 \
  --fold ${SLURM_ARRAY_TASK_ID} \
  --batch_size 16 \
  --epochs 30 \
  --lr 5e-4 \
  --weight_decay 1e-4

echo "END: $(date)"

