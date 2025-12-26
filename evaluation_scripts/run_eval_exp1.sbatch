#!/bin/bash
#SBATCH -J eval_exp1
#SBATCH -p gpu
#SBATCH -A orsdardencomputing
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/eval_exp1.out
#SBATCH -e logs/eval_exp1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zjy6us@virginia.edu

cd /home/zjy6us/mlia/evaluation_scripts

module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

source ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/zjy6us/mlia

python eval_exp1.py
