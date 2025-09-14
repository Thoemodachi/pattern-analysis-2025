#!/bin/bash
#SBATCH --job-name=cifar10-fast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module purge
module load cuda/12.1  # Or the correct CUDA version on Rangpur
# Activate your Python environment here:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyt24

cd ~/code   # <--- go into the code directory

python fast_cifar10.py --epochs 32 --batch 1024 --workers 8