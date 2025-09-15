#!/bin/bash
#SBATCH --job-name=cifar10-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/$USER/slurm_logs/%x-%j.out

# Load CUDA and Conda
module purge
module load cuda/12.1
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your Conda env
conda activate ai

cd ~/dev/pattern-analysis-2025/code

RUN_DIR=/home/$USER/runs/train-${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

python fast_cifar10.py --epochs 60 --batch 128 --workers 8 --lr 0.1
mv cifar10_resnet18_baseline.pt "$RUN_DIR/"
