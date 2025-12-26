import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random

def create_comparison_grid(dataset_path, output_path='input_vs_resized_comparison.png', grid_size=4):
    """
    Creates a side-by-side comparison of original vs resized images.
    """
    # 1. Find images
    exts = ['jpg', 'jpeg', 'png', 'tiff']
    # SORT to ensure deterministic order
    paths = sorted([p for ext in exts for p in Path(dataset_path).glob(f'**/*.{ext}')])
    
    if not paths:
        print(f"No images found in {dataset_path}")
        return

    print(f"Found {len(paths)} images. Selecting {grid_size*grid_size} random samples...")

    # 2. Select random samples
    random.seed(42)  # Use same seed for consistency with the other script
    selected_paths = random.sample(paths, grid_size * grid_size)

    # 3. Create plot
    fig = plt.figure(figsize=(12, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.05)
    
    titles = ['Original Input Data', 'Resized (64x64)']
    
    # Transform for resizing (Right Grid Only)
    resize_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    
    for idx, subfig in enumerate(subfigs):
        subfig.suptitle(titles[idx], fontsize=14, fontweight='bold')
        axs = subfig.subplots(grid_size, grid_size)
        
        for i, ax in enumerate(axs.flat):
            img_path = selected_paths[i]
            try:
                img = Image.open(img_path).convert('RGB')
                
                if idx == 1: # Resized Grid
                    img = resize_transform(img)
                else:
                    # Original Grid: No transform
                    pass
                
                ax.imshow(img)
                ax.axis('off')
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                ax.axis('off')

    # Save without forcing DPI
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved comparison grid to {output_path}")

if __name__ == "__main__":
    create_comparison_grid('./sysu-shape-dataset/combined/', 'input_vs_resized_comparison.png', grid_size=4)
