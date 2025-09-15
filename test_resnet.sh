#!/bin/bash
#SBATCH --job-name=cifar10-test
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/$USER/slurm_logs/%x-%j.out

# --------------------------
# Environment setup
# --------------------------
module purge
module load cuda/12.1
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

# activate your environment (replace ai with your env name)
conda activate ai

# --------------------------
# Run test script
# --------------------------
cd ~/dev/pattern-analysis-2025/code

# Path to trained checkpoint (update jobid as needed)
CKPT=/home/$USER/runs/train-278600/cifar10_resnet18_baseline.pt

python test_cifar10.py \
  --ckpt "$CKPT" \
  --batch 2048 \
  --workers 8 \
  --amp true
