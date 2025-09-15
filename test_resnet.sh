#!/bin/bash
#SBATCH --job-name=cifar10-test
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/$USER/slurm_logs/%x-%j.out

module purge
module load cuda/12.1
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ai

cd ~/dev/pattern-analysis-2025/code

MODEL_PATH=/home/$USER/runs/train-278600/cifar10_resnet18_baseline.pt
python fast_cifar10.py --test_only "$MODEL_PATH"
