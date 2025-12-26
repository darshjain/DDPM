import re
import numpy as np
from pathlib import Path

def analyze_logs():
    log_dir = Path('/home/zjy6us/mlia/logs')
    
    experiments = {
        'Baseline (Dim 64)': 'mliav4-mild-augmented-more-steps.err',
        'Low LR': 'exp1_low_lr.err',
        'Low LR + Aug': 'exp2_hybrid.err',
        'Dim 96 + Aug': 'exp3_deeper.err',
        'Dim 128 + Aug': 'exp4_dim128.err'
    }
    
    pattern = re.compile(r'loss:\s*([0-9.]+):.*\|\s*(\d+)/')
    
    print(f"{'Experiment':<25} | {'Min Loss':<15} | {'Step (Min)':<12} | {'Final Loss (Avg)':<15} | {'Trend'}")
    print("-" * 90)
    
    for name, filename in experiments.items():
        file_path = log_dir / filename
        losses = []
        steps = []
        
        if not file_path.exists():
            continue
            
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    losses.append(float(match.group(1)))
                    steps.append(int(match.group(2)))
        
        if not losses:
            continue
            
        # Analysis
        min_loss = np.min(losses)
        min_idx = np.argmin(losses)
        min_step = steps[min_idx]
        
        # Final average (last 1000 steps) to check stability
        final_avg = np.mean(losses[-1000:])
        
        # Trend analysis
        first_half_avg = np.mean(losses[len(losses)//4:len(losses)//2])
        second_half_avg = np.mean(losses[len(losses)//2:])
        
        trend = "Stable"
        if second_half_avg < first_half_avg * 0.9:
            trend = "Still Improving"
        elif second_half_avg > first_half_avg * 1.1:
            trend = "Diverging/Unstable"
            
        print(f"{name:<25} | {min_loss:.6f}        | {min_step:<12} | {final_avg:.6f}        | {trend}")

if __name__ == '__main__':
    analyze_logs()

