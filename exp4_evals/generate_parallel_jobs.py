import os

# Configuration
start_idx = 17
end_idx = 25
base_path = "/home/zjy6us/mlia/results_exp4_dim128"
eval_utils_path = "/home/zjy6us/mlia/exp4_evals"

# Templates
py_template = """import sys
from pathlib import Path
# Add current directory to path to find eval_utils
sys.path.append("{eval_utils_path}")
from eval_utils import evaluate_checkpoint
import torch
import json

# TARGET MODEL
CHECKPOINT_PATH = '{checkpoint_path}'

def main():
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {{checkpoint_path}}")
        return

    # Configuration
    DIM = 128
    REAL_DATA = '/home/zjy6us/mlia/sysu-shape-dataset/combined/'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating: {{checkpoint_path.name}}")
    print(f"Device: {{device}}")

    # Evaluate
    scores = evaluate_checkpoint(
        str(checkpoint_path), 
        DIM, 
        device, 
        REAL_DATA
    )

    # Calculate step number
    milestone = int(checkpoint_path.stem.split('-')[1])
    step = milestone * 4000
    
    # Save to model-X_eval.json
    output_path = checkpoint_path.parent / f"{{checkpoint_path.stem}}_eval.json"
    
    result_data = {{
        str(step): scores
    }}
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
        
    print(f"✓ Results saved to {{output_path}}")

if __name__ == '__main__':
    main()
"""

sbatch_template = """#!/bin/bash
#SBATCH -J eval_{idx}
#SBATCH -p gpu
#SBATCH -A orsdardencomputing
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_{idx}_%j.out
#SBATCH --error=logs/eval_{idx}_%j.err

cd /home/zjy6us/mlia/exp4_evals

module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

source ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/zjy6us/mlia

python eval_{idx}.py
"""

# Generate files
print("Generating evaluation files...")
for i in range(start_idx, end_idx + 1):
    # File names
    py_filename = f"eval_{i}.py"
    sbatch_filename = f"run_{i}.sbatch"
    checkpoint_path = f"{base_path}/model-{i}.pt"
    
    # Write Python file
    with open(py_filename, "w") as f:
        f.write(py_template.format(
            eval_utils_path=eval_utils_path,
            checkpoint_path=checkpoint_path
        ))
    
    # Write Sbatch file
    with open(sbatch_filename, "w") as f:
        f.write(sbatch_template.format(idx=i))
        
    print(f"Created {py_filename} and {sbatch_filename}")

print("\nTo submit all jobs, run:")
print("for i in {17..25}; do sbatch run_$i.sbatch; done")

