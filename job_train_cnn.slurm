#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/cnn_%j.out

module load python/3.10
source /scratch/lustre/home/$USER/projects/sleep-staging-bsc/.venv/bin/activate
cd /scratch/lustre/home/$USER/projects/sleep-staging-bsc

mkdir -p logs models results

python -m src.sleepstaging.train_sleep_staging

