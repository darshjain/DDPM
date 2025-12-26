#!/bin/bash
#SBATCH -J evaluate_all
#SBATCH -p gpu
#SBATCH -A orsdardencomputing
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/evaluate_all.out
#SBATCH -e logs/evaluate_all.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zjy6us@virginia.edu

cd /home/zjy6us/mlia

module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

source venv/bin/activate

python evaluate_all.py
