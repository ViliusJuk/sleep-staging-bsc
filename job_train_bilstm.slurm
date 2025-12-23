#!/bin/bash
#SBATCH --job-name=bilstm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/bilstm_%j.out

# Jei reikia klasterio profilio:
source ~/.bashrc  # (nebūtina, bet dažnai naudinga)

# Aktyvuok savo virtualenv
source /scratch/lustre/home/$USER/projects/sleep-staging-bsc/.venv/bin/activate

cd /scratch/lustre/home/$USER/projects/sleep-staging-bsc
mkdir -p logs models results

python -m src.sleepstaging.train_bilstm

