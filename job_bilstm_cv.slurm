#!/bin/bash
#SBATCH --job-name=bilstm_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:tesla-v100-sxm2-32gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bilstm_cv_%A_%a.out
#SBATCH --error=logs/bilstm_cv_%A_%a.err

module load python
source .venv/bin/activate

echo "JOB ID: $SLURM_JOB_ID"
echo "FOLD: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi || true

python -u -m src.sleepstaging.train_bilstm_cv \
  --k 10 \
  --fold ${SLURM_ARRAY_TASK_ID}

echo "END: $(date)"

