#!/bin/bash
#SBATCH --job-name=cifar10-fast
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/$USER/slurm_logs/%x-%j.out

module purge
module load cuda/12.1
cd ~/dev/pattern-analysis-2025/code
python fast_cifar10.py --epochs 32 --batch 1024 --workers 8 --out_dir /home/$USER/runs