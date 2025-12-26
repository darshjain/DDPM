import sys
from pathlib import Path
from eval_utils import evaluate_checkpoint
import torch
import json

# ==========================================
# 👇 CHANGE THIS PATH FOR EACH RUN
# ==========================================
CHECKPOINT_PATH = '/home/zjy6us/mlia/results_exp4_dim128/model-25.pt'
# ==========================================

def main():
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Configuration
    DIM = 128  # Model dimension (128 for Exp 4)
    REAL_DATA = '/home/zjy6us/mlia/sysu-shape-dataset/combined/'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"Device: {device}")

    # Evaluate
    scores = evaluate_checkpoint(
        str(checkpoint_path), 
        DIM, 
        device, 
        REAL_DATA
    )

    # Calculate step number (model-X.pt -> X * 4000)
    milestone = int(checkpoint_path.stem.split('-')[1])
    step = milestone * 4000
    
    # Save to a separate JSON file: model-X_eval.json
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_eval.json"
    
    result_data = {
        str(step): scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
        
    print(f"✓ Results saved to {output_path}")

if __name__ == '__main__':
    main()

