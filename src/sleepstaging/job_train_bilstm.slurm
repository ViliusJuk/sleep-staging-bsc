#!/bin/bash
#SBATCH --job-name=bilstm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/bilstm_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load python/3.10

# aktyvuojam env
source ~/.venv/bin/activate || source sleep-staging-bsc/.venv/bin/activate

cd /scratch/lustre/home/viju8335/projects/sleep-staging-bsc

python -u -m src.sleepstaging.train_ma_bilstm

