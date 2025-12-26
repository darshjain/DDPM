#!/usr/bin/env python3
"""
Create a collage of 64 random input images from the dataset
in the same 8x8 grid format as the generated samples.
"""

import torch
from torchvision import utils
from pathlib import Path
from PIL import Image
import random
from torchvision import transforms

# Configuration
DATASET_FOLDER = './sysu-shape-dataset/combined/'
OUTPUT_FILE = 'input_collage_64.png'
NUM_IMAGES = 64
IMAGE_SIZE = 64
GRID_SIZE = 8  # 8x8 grid

def main():
    # Find all images in the dataset
    exts = ['jpg', 'jpeg', 'png', 'tiff']
    all_paths = []
    for ext in exts:
        all_paths.extend(Path(DATASET_FOLDER).glob(f'**/*.{ext}'))
    
    print(f"Found {len(all_paths)} images in dataset")
    
    if len(all_paths) < NUM_IMAGES:
        print(f"Warning: Dataset has only {len(all_paths)} images, but {NUM_IMAGES} requested")
        print(f"Will use all available images")
        selected_paths = all_paths
    else:
        # Randomly sample 64 images
        selected_paths = random.sample(all_paths, NUM_IMAGES)
    
    # Transform to resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Load and transform images
    images = []
    for path in selected_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    # Stack into a batch
    images_batch = torch.stack(images)
    
    print(f"Created batch of {len(images)} images with shape {images_batch.shape}")
    
    # Save as grid (same format as the generated samples)
    utils.save_image(
        images_batch, 
        OUTPUT_FILE,
        nrow=GRID_SIZE,
        normalize=False  # Images are already in [0, 1] range
    )
    
    print(f"✓ Saved input collage to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
