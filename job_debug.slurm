#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/debug_%j.out
#SBATCH --error=/scratch/lustre/home/%u/projects/sleep-staging-bsc/logs/debug_%j.err

cd /scratch/lustre/home/viju8335/projects/sleep-staging-bsc

echo "=== BEFORE VENV ==="
which python
python -V

echo "=== ACTIVATING VENV ==="
source /scratch/lustre/home/viju8335/projects/sleep-staging-bsc/.venv/bin/activate

echo "=== AFTER VENV ==="
which python
python -V

echo "=== TRY IMPORT TRAIN ==="
python - << 'EOF'
try:
    import src.sleepstaging.train_ma_bilstm
    print("IMPORT OK ✅")
except Exception as e:
    print("IMPORT FAILED ❌", e)
EOF

echo "=== DONE ==="

