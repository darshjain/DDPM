import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import math
import random

def create_input_collage(dataset_path, output_path='input_dataset_sample_4x4.png', grid_size=4):
    """
    Creates a grid of sample images from the input dataset.
    """
    # 1. Find images
    exts = ['jpg', 'jpeg', 'png', 'tiff']
    # SORT to ensure deterministic order across runs
    paths = sorted([p for ext in exts for p in Path(dataset_path).glob(f'**/*.{ext}')])
    
    if not paths:
        print(f"No images found in {dataset_path}")
        return

    print(f"Found {len(paths)} images. Selecting {grid_size*grid_size} random samples...")

    # 2. Select random samples
    # Set seed for reproducibility (and to match the other script)
    random.seed(42)
    selected_paths = random.sample(paths, grid_size * grid_size)

    # 3. Create plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, ax in enumerate(axes.flat):
        img_path = selected_paths[i]
        try:
            # Load original image WITHOUT resizing
            img = Image.open(img_path).convert('RGB')
            # Plot raw image - matplotlib will scale to fit the axis
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            ax.axis('off')

    # Save without forcing DPI (uses default)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved dataset sample grid to {output_path}")

if __name__ == "__main__":
    create_input_collage('./sysu-shape-dataset/combined/', 'input_dataset_sample_4x4.png', grid_size=4)
