#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/transformer_%j.out

# priklausomai nuo tavo klasterio gali nereikėti module load
# module load python/3.10   # jei neveikia, palik iškomentuotą

source /scratch/lustre/home/$USER/projects/sleep-staging-bsc/.venv/bin/activate
cd /scratch/lustre/home/$USER/projects/sleep-staging-bsc

mkdir -p logs models results

python -m src.sleepstaging.train_transformer

