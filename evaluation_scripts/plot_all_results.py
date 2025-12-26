import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_combined_results():
    """Generate combined and individual plots from JSON results."""
    print("Generating comparison plots...")
    
    # Define where to look for results
    # Mapped to professional names for the report
    experiments = {
        'Baseline (Dim 64)': '/home/zjy6us/mlia/results_combinedV4-mild-augmented-more-steps',
        'Low LR': '/home/zjy6us/mlia/results_exp1_low_lr_no_aug',
        'Low LR + Aug': '/home/zjy6us/mlia/results_exp2_low_lr_mild_aug',
        'Dim 96 + Aug': '/home/zjy6us/mlia/results_exp3_deeper_model',
        'Dim 128 + Aug': '/home/zjy6us/mlia/results_exp4_dim128'
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    all_data = {}
    
    # Load all data first
    for name, path in experiments.items():
        json_path = Path(path) / 'evaluation_results.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                all_data[name] = json.load(f)
        else:
            print(f"Warning: No results found for {name} at {json_path}")

    # 1. Combined FID Plot
    plt.figure(figsize=(12, 8))
    for (name, _), color in zip(experiments.items(), colors):
        if name not in all_data: continue
        data = all_data[name]
        steps = sorted([int(k) for k in data.keys()])
        fids = [data[str(s)]['fid_score'] for s in steps]
        valid = [(s, f) for s, f in zip(steps, fids) if f is not None]
        if valid:
            x, y = zip(*valid)
            plt.plot(x, y, marker='o', linewidth=2, label=name, color=color)
            
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('FID Score (Lower is Better)', fontsize=12)
    plt.title('DDPM Training: FID Score Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('comparison_fid.png', dpi=300)
    
    # 2. Combined IS Plot
    plt.figure(figsize=(12, 8))
    for (name, _), color in zip(experiments.items(), colors):
        if name not in all_data: continue
        data = all_data[name]
        steps = sorted([int(k) for k in data.keys()])
        iss = [data[str(s)]['is_mean'] for s in steps]
        valid = [(s, i) for s, i in zip(steps, iss) if i is not None]
        if valid:
            x, y = zip(*valid)
            plt.plot(x, y, marker='o', linewidth=2, label=name, color=color)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Inception Score (Higher is Better)', fontsize=12)
    plt.title('DDPM Training: Inception Score Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('comparison_is.png', dpi=300)
    
    print("✓ Combined plots saved.")
    
    # 3. Individual Plots
    print("Generating individual plots...")
    for (name, _) in experiments.items():
        if name not in all_data: continue
        
        # Clean name for filename
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        data = all_data[name]
        steps = sorted([int(k) for k in data.keys()])
        fids = [data[str(s)]['fid_score'] for s in steps]
        iss = [data[str(s)]['is_mean'] for s in steps]
        is_stds = [data[str(s)]['is_std'] for s in steps]
        
        # Individual FID
        plt.figure(figsize=(10, 6))
        valid_fid = [(s, f) for s, f in zip(steps, fids) if f is not None]
        if valid_fid:
            x, y = zip(*valid_fid)
            plt.plot(x, y, marker='o', linewidth=2, color='#2E86AB')
            best_idx = np.argmin(y)
            plt.plot(x[best_idx], y[best_idx], 'r*', markersize=15, 
                    label=f'Best: {y[best_idx]:.4f} at step {x[best_idx]}')
            
        plt.xlabel('Training Steps')
        plt.ylabel('FID Score (Lower is Better)')
        plt.title(f'FID Score - {name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'fid_{safe_name}.png', dpi=300)
        plt.close()
        
        # Individual IS
        plt.figure(figsize=(10, 6))
        valid_is = [(s, m, std) for s, m, std in zip(steps, iss, is_stds) if m is not None]
        if valid_is:
            x, y, err = zip(*valid_is)
            plt.errorbar(x, y, yerr=err, marker='o', linewidth=2, capsize=5, color='#A23B72')
            best_idx = np.argmax(y)
            plt.plot(x[best_idx], y[best_idx], 'r*', markersize=15,
                    label=f'Best: {y[best_idx]:.4f} at step {x[best_idx]}')
            
        plt.xlabel('Training Steps')
        plt.ylabel('Inception Score (Higher is Better)')
        plt.title(f'Inception Score - {name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'is_{safe_name}.png', dpi=300)
        plt.close()
        
    print("✓ Individual plots saved.")

if __name__ == '__main__':
    plot_combined_results()

