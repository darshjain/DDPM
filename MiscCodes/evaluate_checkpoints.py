"""
evaluate_checkpoints.py
=======================

Comprehensive evaluation script for trained diffusion model checkpoints.

Purpose:
--------
Evaluates the quality of generated images from diffusion model checkpoints using
two standard generative model metrics:
  1. FID (Fréchet Inception Distance) - measures similarity to real images (lower is better)
  2. IS (Inception Score) - measures quality and diversity (higher is better)

What it does:
-------------
- Loads all model checkpoints from a results folder
- For each checkpoint:
  * Generates 2,500 sample images
  * Calculates FID score by comparing to real dataset
  * Calculates Inception Score for quality/diversity
- Saves all results to 'evaluation_results.json'
- Creates visualization plots showing metrics across training steps
- Identifies and highlights best performing checkpoints

Usage:
------
  python evaluate_checkpoints.py

Requirements:
-------------
- GPU recommended (much faster than CPU)
- Real dataset at './sysu-shape-dataset/combined/'
- Checkpoints in './results_combinedV3-withoutaugmentation/'

Output:
-------
- evaluation_results.json: All metric scores
- evaluation_curves.png: Visualization plots
- Console output with summary table and best checkpoints

Note: This script takes significant time to run as it generates thousands of
      images per checkpoint. Results are cached in JSON for later re-plotting.
"""

import torch
import numpy as np
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from scipy.stats import entropy
from torchvision.models import inception_v3
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ========== Inception Score Calculator ==========
class InceptionScore:
    def __init__(self, device):
        self.device = device  # Use GPU for much faster inference!
        self.model = None

    def load_model(self):
        if self.model is None:
            try:
                from torchvision.models import Inception_V3_Weights
                weights = Inception_V3_Weights.DEFAULT
                self.model = inception_v3(weights=weights).to(self.device)
            except (ImportError, AttributeError):
                self.model = inception_v3(pretrained=True).to(self.device)
            self.model.eval()

    def calculate(self, images, splits=10):
        """
        images: (N, 3, H, W) tensor, normalized [0, 1]
        """
        self.load_model()
        images = images.to(self.device)  # Keep on GPU for speed

        N = len(images)
        # Resize to 299x299 which is required for InceptionV3
        up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        
        preds = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = images[i:i + batch_size]
                batch = up(batch)
                # Normalize: (batch - mean) / std
                # ImageNet mean and std
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                batch = (batch - mean) / std
                
                output = self.model(batch)
                preds.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())
                
        preds = np.concatenate(preds, axis=0)
        
        split_scores = []
        chunk_size = N // splits
        if chunk_size == 0:
             splits = 1
             chunk_size = N
             
        for k in range(splits):
            part = preds[k * chunk_size : (k + 1) * chunk_size, :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            
        return np.mean(split_scores), np.std(split_scores)


def num_to_groups(num, divisor):
    """Split num into groups of divisor size."""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class SimpleDataset(Dataset):
    """Simple dataset for loading images."""
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)


def evaluate_checkpoint(checkpoint_path, device, real_data_path, num_samples=2500):
    """
    Evaluate a single checkpoint and return FID and IS.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        dict with 'fid_score', 'is_mean', 'is_std'
    """
    # Load model
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        flash_attn=False
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise'
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    diffusion.eval()
    
    results = {}
    
    # Create data loader for real images
    print(f"  Loading real dataset from {real_data_path}...")
    dataset = SimpleDataset(real_data_path, image_size=64)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    # Convert to iterator - FIDEvaluation expects an iterator, not a DataLoader
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    dataloader_iter = cycle(dataloader)
    print(f"    Found {len(dataset)} real images")
    
    # Generate samples once and reuse for both FID and IS
    print(f"  Generating {num_samples} samples for evaluation...")
    all_images = []
    batch_size = 64
    batches = num_to_groups(num_samples, batch_size)
    
    try:
        with torch.inference_mode():
            for i, batch_num in enumerate(batches):
                print(f"    Batch {i+1}/{len(batches)} ({batch_num} samples)...", end='\r')
                samples = diffusion.sample(batch_size=batch_num)
                all_images.append(samples)
        print()  # New line after progress
        
        all_images = torch.cat(all_images, dim=0)
        print(f"    ✓ Generated {len(all_images)} samples")
    except Exception as e:
        print(f"    ✗ Sample generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'fid_score': None, 'is_mean': None, 'is_std': None}
    
    results = {}
    
    # Calculate FID using the generated samples
    print(f"  Calculating FID...")
    try:
        # Create fid_stats directory if it doesn't exist
        Path('./fid_stats').mkdir(parents=True, exist_ok=True)
        
        fid_evaluator = FIDEvaluation(
            batch_size=128,
            dl=dataloader_iter,
            sampler=diffusion,
            channels=3,
            accelerator=None,
            stats_dir='./fid_stats',
            device=device,
            num_fid_samples=num_samples,
            inception_block_idx=2048
        )
        
        # Use pre-generated samples instead of generating new ones
        # We need to monkey-patch the sample method to return our pre-generated samples
        samples_iter = iter([all_images[i:i+128] for i in range(0, len(all_images), 128)])
        original_sample = diffusion.sample
        diffusion.sample = lambda batch_size: next(samples_iter)
        
        fid_score = fid_evaluator.fid_score()
        
        # Restore original sample method
        diffusion.sample = original_sample
        
        results['fid_score'] = fid_score
        print(f"    ✓ FID: {fid_score:.4f}")
    except Exception as e:
        print(f"    ✗ FID calculation failed: {e}")
        import traceback
        traceback.print_exc()
        results['fid_score'] = None
    
    # Calculate Inception Score using the same generated samples
    print(f"  Calculating Inception Score...")
    try:
        inception_scorer = InceptionScore(device=device)
        imgs_for_is = all_images.clamp(0, 1)
        
        is_mean, is_std = inception_scorer.calculate(imgs_for_is)
        results['is_mean'] = is_mean
        results['is_std'] = is_std
        print(f"    ✓ IS: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"    Inception Score calculation failed: {e}")
        results['is_mean'] = None
        results['is_std'] = None
    
    return results


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    results_folder = Path('./results_combinedV3-withoutaugmentation')
    checkpoint_files = sorted(results_folder.glob('model-*.pt'), 
                            key=lambda x: int(x.stem.split('-')[1]))
    
    print(f"\nFound {len(checkpoint_files)} checkpoints")
    print("="*70)
    
    # Real data path
    real_data_path = './sysu-shape-dataset/combined/'
    
    # Evaluate each checkpoint
    evaluation_results = {}
    
    for checkpoint_file in checkpoint_files:
        milestone = int(checkpoint_file.stem.split('-')[1])
        epoch = milestone * 2000  # Since save_and_sample_every=2000
        
        print(f"\nEvaluating Milestone {milestone} (Step {epoch})...")
        
        try:
            results = evaluate_checkpoint(checkpoint_file, device, real_data_path, num_samples=2500)
            evaluation_results[epoch] = results
        except Exception as e:
            print(f"  Failed to evaluate: {e}")
            evaluation_results[epoch] = {'fid_score': None, 'is_mean': None, 'is_std': None}
    
    # Save results to JSON
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for epoch, results in evaluation_results.items():
            serializable_results[str(epoch)] = {
                'fid_score': float(results['fid_score']) if results['fid_score'] is not None else None,
                'is_mean': float(results['is_mean']) if results['is_mean'] is not None else None,
                'is_std': float(results['is_std']) if results['is_std'] is not None else None
            }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Plot results
    print("\nGenerating plots...")
    epochs = sorted([int(e) for e in evaluation_results.keys()])
    fid_scores = [evaluation_results[e]['fid_score'] for e in epochs]
    is_means = [evaluation_results[e]['is_mean'] for e in epochs]
    is_stds = [evaluation_results[e]['is_std'] for e in epochs]
    
    # Filter out None values
    valid_fid = [(e, f) for e, f in zip(epochs, fid_scores) if f is not None]
    valid_is = [(e, m, s) for e, m, s in zip(epochs, is_means, is_stds) if m is not None]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # FID plot
    if valid_fid:
        fid_epochs, fid_vals = zip(*valid_fid)
        ax1.plot(fid_epochs, fid_vals, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('FID Score (lower is better)', fontsize=12)
        ax1.set_title('Fréchet Inception Distance vs Training Steps', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Find best FID
        best_fid_idx = np.argmin(fid_vals)
        best_fid_epoch = fid_epochs[best_fid_idx]
        best_fid_score = fid_vals[best_fid_idx]
        ax1.plot(best_fid_epoch, best_fid_score, 'r*', markersize=20, 
                label=f'Best: {best_fid_score:.4f} at step {best_fid_epoch}')
        ax1.legend(fontsize=10)
        print(f"\n🏆 Best FID Score: {best_fid_score:.4f} at step {best_fid_epoch}")
    
    # IS plot
    if valid_is:
        is_epochs, is_vals, is_err = zip(*valid_is)
        ax2.errorbar(is_epochs, is_vals, yerr=is_err, marker='o', linewidth=2, 
                    markersize=8, capsize=5)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Inception Score (higher is better)', fontsize=12)
        ax2.set_title('Inception Score vs Training Steps', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Find best IS
        best_is_idx = np.argmax(is_vals)
        best_is_epoch = is_epochs[best_is_idx]
        best_is_score = is_vals[best_is_idx]
        best_is_std = is_err[best_is_idx]
        ax2.plot(best_is_epoch, best_is_score, 'r*', markersize=20,
                label=f'Best: {best_is_score:.4f}±{best_is_std:.4f} at step {best_is_epoch}')
        ax2.legend(fontsize=10)
        print(f"🏆 Best IS Score: {best_is_score:.4f}±{best_is_std:.4f} at step {best_is_epoch}")
    
    plt.tight_layout()
    plt.savefig('evaluation_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to evaluation_curves.png")
    
    # Print summary table
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Step':<10} {'FID Score':<15} {'Inception Score':<20}")
    print("-"*70)
    for epoch in epochs:
        fid = evaluation_results[epoch]['fid_score']
        is_mean = evaluation_results[epoch]['is_mean']
        is_std = evaluation_results[epoch]['is_std']
        
        fid_str = f"{fid:.4f}" if fid is not None else "N/A"
        is_str = f"{is_mean:.4f}±{is_std:.4f}" if is_mean is not None else "N/A"
        
        print(f"{epoch:<10} {fid_str:<15} {is_str:<20}")
    print("="*70)


if __name__ == '__main__':
    main()
