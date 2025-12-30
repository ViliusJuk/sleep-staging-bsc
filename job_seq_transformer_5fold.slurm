#!/bin/bash
#SBATCH --job-name=seqtr_5fold
#SBATCH --partition=gpu
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:tesla-v100-sxm2-32gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/seqtr_%A_%a.out
#SBATCH --error=logs/seqtr_%A_%a.err

module load python
source .venv/bin/activate

echo "JOB ID: $SLURM_JOB_ID"
echo "FOLD: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi || true

python -u -m src.sleepstaging.train_seq_transformer_cv \
  --k 10 \
  --fold ${SLURM_ARRAY_TASK_ID} \
  --seq_len 21 \
  --stride 1 \
  --batch_size 64 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers 2 \
  --dim_feedforward 256 \
  --dropout 0.1

echo "END: $(date)"

