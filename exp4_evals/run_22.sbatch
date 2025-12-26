#!/bin/bash
#SBATCH -J eval_22
#SBATCH -p gpu
#SBATCH -A orsdardencomputing
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_22_%j.out
#SBATCH --error=logs/eval_22_%j.err

cd /home/zjy6us/mlia/exp4_evals

module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

source ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/zjy6us/mlia

python eval_22.py
