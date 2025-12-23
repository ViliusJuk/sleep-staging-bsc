#!/bin/bash
#SBATCH --job-name=cnn_cv10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-9
#SBATCH --output=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/cnn_cv10_%A_%a.out

source /scratch/lustre/home/$USER/projects/sleep-staging-bsc/.venv/bin/activate
cd /scratch/lustre/home/$USER/projects/sleep-staging-bsc

mkdir -p logs models results

python -m src.sleepstaging.train_cnn_cv --k 10 --fold ${SLURM_ARRAY_TASK_ID}

