#!/bin/bash
#SBATCH --job-name=cifar10-fast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/$USER/slurm_logs/%x-%j.out   # SLURM file

module purge
module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate ai

cd ~/code
python fast_cifar10.py --epochs 32 --batch 1024 --workers 8 --out_dir /home/$USER/runs