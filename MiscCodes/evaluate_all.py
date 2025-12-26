"""
evaluate_all.py
===============

Master evaluation script for comparing multiple diffusion model experiments.

Purpose:
--------
Evaluates all checkpoints from 4 specific experiments:
1. Original (mliav4)
2. Exp 1 (Low LR)
3. Exp 2 (Hybrid)
4. Exp 3 (Deeper Model)

What it does:
-------------
- Scans each experiment's result folder for checkpoints
- Calculates FID and IS for EVERY checkpoint
- Saves results incrementally (so progress isn't lost)
- Generates comprehensive comparison plots:
  * Combined FID Comparison
  * Combined IS Comparison
  * Individual plots for each experiment

Usage:
------
  python evaluate_all.py

Requirements:
-------------
- GPU (Essential for speed)
- Real dataset at './sysu-shape-dataset/combined/'
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
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys

# ========== Inception Score Calculator ==========
class InceptionScore:
    def __init__(self, device):
        self.device = device
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
        self.load_model()
        images = images.to(self.device)
        N = len(images)
        up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        preds = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = images[i:i + batch_size]
                batch = up(batch)
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
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class SimpleDataset(Dataset):
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

def evaluate_checkpoint(checkpoint_path, model_dim, device, real_data_path, num_samples=2500):
    """Evaluate a single checkpoint."""
    # Load model with correct dimensions based on experiment
    model = Unet(
        dim=model_dim,
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
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats (EMA vs Model)
        if 'ema' in checkpoint:
            print("    Loading EMA weights...")
            # Create EMA wrapper to load weights correctly
            from ema_pytorch import EMA
            ema = EMA(diffusion, beta=0.995, update_every=10)
            ema.load_state_dict(checkpoint['ema'])
            diffusion_to_sample = ema.ema_model
        else:
            print("    Loading standard weights...")
            diffusion.load_state_dict(checkpoint['model'])
            diffusion_to_sample = diffusion
            
        diffusion_to_sample.eval()
    except Exception as e:
        print(f"    Error loading checkpoint: {e}")
        return {'fid_score': None, 'is_mean': None, 'is_std': None}
    
    # Real data loader
    dataset = SimpleDataset(real_data_path, image_size=64)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    dataloader_iter = cycle(dataloader)
    
    # Generate samples
    print(f"    Generating {num_samples} samples...")
    all_images = []
    batch_size = 64
    batches = num_to_groups(num_samples, batch_size)
    
    try:
        with torch.inference_mode():
            for i, batch_num in enumerate(batches):
                samples = diffusion_to_sample.sample(batch_size=batch_num)
                all_images.append(samples)
        all_images = torch.cat(all_images, dim=0)
    except Exception as e:
        print(f"    Generation failed: {e}")
        return {'fid_score': None, 'is_mean': None, 'is_std': None}
    
    results = {}
    
    # FID
    print(f"    Calculating FID...")
    try:
        Path('./fid_stats').mkdir(parents=True, exist_ok=True)
        fid_evaluator = FIDEvaluation(
            batch_size=128,
            dl=dataloader_iter,
            sampler=diffusion_to_sample,
            channels=3,
            accelerator=None,
            stats_dir='./fid_stats',
            device=device,
            num_fid_samples=num_samples,
            inception_block_idx=2048
        )
        
        # Monkey patch sampling
        samples_iter = iter([all_images[i:i+128] for i in range(0, len(all_images), 128)])
        original_sample = diffusion_to_sample.sample
        diffusion_to_sample.sample = lambda batch_size: next(samples_iter)
        
        fid_score = fid_evaluator.fid_score()
        diffusion_to_sample.sample = original_sample
        results['fid_score'] = fid_score
        print(f"    ✓ FID: {fid_score:.4f}")
    except Exception as e:
        print(f"    FID failed: {e}")
        results['fid_score'] = None
    
    # IS
    print(f"    Calculating IS...")
    try:
        inception_scorer = InceptionScore(device=device)
        imgs_for_is = all_images.clamp(0, 1)
        is_mean, is_std = inception_scorer.calculate(imgs_for_is)
        results['is_mean'] = is_mean
        results['is_std'] = is_std
        print(f"    ✓ IS: {is_mean:.4f}")
    except Exception as e:
        print(f"    IS failed: {e}")
        results['is_mean'] = None
        results['is_std'] = None
        
    return results

def plot_combined_results(all_results):
    """Generate combined and individual plots."""
    print("\nGenerating comparison plots...")
    
    experiments = {
        'Original': 'results_combinedV4-mild-augmented-more-steps',
        'Exp 1 (Low LR)': 'results_exp1_low_lr_no_aug',
        'Exp 2 (Hybrid)': 'results_exp2_low_lr_mild_aug',
        'Exp 3 (Deeper)': 'results_exp3_deeper_model'
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Combined FID Plot
    plt.figure(figsize=(12, 8))
    for (name, folder), color in zip(experiments.items(), colors):
        if name not in all_results: continue
        data = all_results[name]
        steps = sorted([int(k) for k in data.keys()])
        fids = [data[str(s)]['fid_score'] for s in steps]
        valid = [(s, f) for s, f in zip(steps, fids) if f is not None]
        if valid:
            x, y = zip(*valid)
            plt.plot(x, y, marker='o', linewidth=2, label=name, color=color)
            
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('FID Score (Lower is Better)', fontsize=12)
    plt.title('FID Score Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('comparison_fid.png', dpi=300)
    
    # 2. Combined IS Plot
    plt.figure(figsize=(12, 8))
    for (name, folder), color in zip(experiments.items(), colors):
        if name not in all_results: continue
        data = all_results[name]
        steps = sorted([int(k) for k in data.keys()])
        iss = [data[str(s)]['is_mean'] for s in steps]
        valid = [(s, i) for s, i in zip(steps, iss) if i is not None]
        if valid:
            x, y = zip(*valid)
            plt.plot(x, y, marker='o', linewidth=2, label=name, color=color)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Inception Score (Higher is Better)', fontsize=12)
    plt.title('Inception Score Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('comparison_is.png', dpi=300)
    
    print("✓ Plots saved to comparison_fid.png and comparison_is.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    real_data_path = './sysu-shape-dataset/combined/'
    
    # Define experiments
    experiments = [
        {'name': 'Original', 'folder': 'results_combinedV4-mild-augmented-more-steps', 'dim': 64},
        {'name': 'Exp 1 (Low LR)', 'folder': 'results_exp1_low_lr_no_aug', 'dim': 64},
        {'name': 'Exp 2 (Hybrid)', 'folder': 'results_exp2_low_lr_mild_aug', 'dim': 64},
        {'name': 'Exp 3 (Deeper)', 'folder': 'results_exp3_deeper_model', 'dim': 96}
    ]
    
    all_results = {}
    
    for exp in experiments:
        name = exp['name']
        folder = Path(exp['folder'])
        dim = exp['dim']
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")
        
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue
            
        # Load existing results if any
        json_path = folder / 'evaluation_results.json'
        exp_results = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    exp_results = json.load(f)
                print(f"Loaded {len(exp_results)} existing results.")
            except:
                pass
        
        checkpoints = sorted(folder.glob('model-*.pt'), 
                           key=lambda x: int(x.stem.split('-')[1]))
        
        for cp in checkpoints:
            milestone = int(cp.stem.split('-')[1])
            # Determine step count based on experiment
            if 'exp' in str(folder):
                step = milestone * 4000
            else:
                step = milestone * 2000
                
            if str(step) in exp_results and exp_results[str(step)]['fid_score'] is not None:
                print(f"Skipping Step {step} (already done)")
                continue
                
            print(f"\nProcessing Step {step}...")
            scores = evaluate_checkpoint(cp, dim, device, real_data_path)
            exp_results[str(step)] = scores
            
            # Save immediately
            with open(json_path, 'w') as f:
                json.dump(exp_results, f, indent=2)
                
        all_results[name] = exp_results
        
    # Generate Plots
    plot_combined_results(all_results)

if __name__ == '__main__':
    main()

