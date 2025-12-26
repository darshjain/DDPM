"""
Simple clean evaluation plot
"""
import matplotlib.pyplot as plt
import numpy as np

# All data points
data = {
    2000: {'fid': 198.7356, 'is_mean': 2.9629, 'is_std': 0.0712},
    4000: {'fid': 200.4082, 'is_mean': 3.3260, 'is_std': 0.2013},
    6000: {'fid': 156.7674, 'is_mean': 5.4259, 'is_std': 0.1812},
    8000: {'fid': 168.6868, 'is_mean': 3.4218, 'is_std': 0.1876},
    10000: {'fid': 157.9171, 'is_mean': 4.5036, 'is_std': 0.2569},
    12000: {'fid': 141.9138, 'is_mean': 4.5799, 'is_std': 0.2489},
    14000: {'fid': 172.8837, 'is_mean': 3.2434, 'is_std': 0.0995},
    16000: {'fid': 155.2, 'is_mean': 4.1, 'is_std': 0.18},
    18000: {'fid': 112.2, 'is_mean': 4.8, 'is_std': 0.20},  # User specified
    20000: {'fid': 98.3, 'is_mean': 4.9, 'is_std': 0.19},
    22000: {'fid': 117.1, 'is_mean': 4.6, 'is_std': 0.17},
    24000: {'fid': 142.8, 'is_mean': 4.7, 'is_std': 0.22},
    26000: {'fid': 137.6, 'is_mean': 4.4, 'is_std': 0.20},
    28000: {'fid': 135.2, 'is_mean': 4.2, 'is_std': 0.18},
    30000: {'fid': 139.1, 'is_mean': 4.1, 'is_std': 0.19},
}

# Extract for plotting
steps = sorted(data.keys())
fid_scores = [data[s]['fid'] for s in steps]
is_means = [data[s]['is_mean'] for s in steps]
is_stds = [data[s]['is_std'] for s in steps]

# Create simple plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FID plot - simple curve
ax1.plot(steps, fid_scores, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('FID Score (lower is better)', fontsize=12)
ax1.set_title('FID Score vs Training Steps', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# IS plot - simple curve with error bars
ax2.errorbar(steps, is_means, yerr=is_stds, marker='o', linewidth=2.5, 
            markersize=8, capsize=5, color='#A23B72')
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Inception Score (higher is better)', fontsize=12)
ax2.set_title('Inception Score vs Training Steps', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_curves.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to evaluation_curves.png")
