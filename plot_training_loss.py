import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def parse_log_file(file_path):
    """Extracts (step, loss) pairs from a log file."""
    steps = []
    losses = []
    
    # Regex to capture "loss: 1.2225: ... | 123/100000"
    # Matches: "loss: <float>: ... | <step>/<total>"
    pattern = re.compile(r'loss:\s*([0-9.]+):.*\|\s*(\d+)/')
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    loss_val = float(match.group(1))
                    step_val = int(match.group(2))
                    
                    # Avoid duplicates (tqdm updates same line multiple times)
                    if not steps or step_val > steps[-1]:
                        steps.append(step_val)
                        losses.append(loss_val)
    except FileNotFoundError:
        print(f"Warning: Log file not found: {file_path}")
        return [], []
        
    return steps, losses

def smooth_data(y, window_size=100):
    """Applies a simple moving average to smooth noisy loss data."""
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def plot_losses():
    # 1. Setup
    log_dir = Path('/home/zjy6us/mlia/logs')
    output_dir = Path('/home/zjy6us/mlia/loss_plots')
    output_dir.mkdir(exist_ok=True)
    
    experiments = {
        'Baseline (Dim 64)': 'mliav4-mild-augmented-more-steps.err',
        'Low LR': 'exp1_low_lr.err',
        'Low LR + Aug': 'exp2_hybrid.err',
        'Dim 96 + Aug': 'exp3_deeper.err',
        'Dim 128 + Aug': 'exp4_dim128.err'
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    
    all_data = {}
    
    # 2. Parse Data
    print("Parsing log files...")
    for (name, filename), color in zip(experiments.items(), colors):
        file_path = log_dir / filename
        steps, losses = parse_log_file(file_path)
        
        if steps:
            print(f"  {name}: Found {len(steps)} data points")
            all_data[name] = {'steps': steps, 'losses': losses, 'color': color}
            
            # 3. Generate Individual Plot
            plt.figure(figsize=(10, 6))
            # Plot raw data faintly
            plt.plot(steps, losses, alpha=0.15, color=color, label='Raw Loss')
            
            # Plot smoothed data
            smoothed_losses = smooth_data(losses, window_size=500)
            # Adjust steps to match smoothed length (convolution shrinks array)
            valid_steps = steps[len(steps)-len(smoothed_losses):]
            
            plt.plot(valid_steps, smoothed_losses, linewidth=2, color=color, label='Smoothed (MA=500)')
            
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title(f'Training Loss: {name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
            output_path = output_dir / f'loss_{safe_name}.png'
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            print(f"  {name}: No data found!")

    # 4. Generate Combined Plot
    print("Generating combined comparison plot...")
    plt.figure(figsize=(12, 8))
    
    for name, data in all_data.items():
        steps = data['steps']
        losses = data['losses']
        color = data['color']
        
        # Only plot smoothed data for comparison (raw is too messy)
        smoothed_losses = smooth_data(losses, window_size=1000) # Higher smoothing for combined plot
        valid_steps = steps[len(steps)-len(smoothed_losses):]
        
        plt.plot(valid_steps, smoothed_losses, linewidth=2, color=color, label=name)
        
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss (Smoothed)', fontsize=12)
    plt.title('Training Loss Comparison (All Experiments)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    combined_path = output_dir / 'comparison_loss.png'
    plt.savefig(combined_path, dpi=300)
    plt.close()
    
    print(f"✓ All plots saved to {output_dir}/")

if __name__ == '__main__':
    plot_losses()

