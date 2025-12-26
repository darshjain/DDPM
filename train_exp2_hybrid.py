import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import time
import sys
from gpu_monitor import GPUMemoryMonitor, print_gpu_info
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from torch import nn
from torchvision.models import inception_v3
from scipy.stats import entropy

# ---------- device selection ----------
def get_device():
    """Select CUDA GPU if available, otherwise exit."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        print("\n" + "="*70)
        print("ERROR: No CUDA GPU detected!")
        sys.exit(1)

# ========== Inception Score Calculator ==========
class InceptionScore:
    def __init__(self, device):
        self.device = torch.device('cpu')  # Force CPU for stability
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

    def calculate(self, images, splits=1):
        self.load_model()
        images = images.cpu()
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

# ========== Augmented Dataset (Mild Augmentation for Exp 2) ==========
class DatasetMildAug(Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        # EXPERIMENT 2: MILD AUGMENTATION
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5), # Keep H-Flip
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

def cycle(dl):
    while True:
        for data in dl:
            yield data

# ========== Enhanced Trainer ==========
class MonitoredTrainer:
    def __init__(self, trainer, memory_monitor):
        self.trainer = trainer
        self.memory_monitor = memory_monitor
        self.start_time = None
        self.inception_scorer = InceptionScore(torch.device('cpu'))
        self._patch_trainer()

    def _patch_trainer(self):
        def patched_train():
            accelerator = self.trainer.accelerator
            device = accelerator.device
            from tqdm import tqdm
            with tqdm(initial=self.trainer.step, total=self.trainer.train_num_steps,
                     disable=not accelerator.is_main_process) as pbar:
                while self.trainer.step < self.trainer.train_num_steps:
                    self.trainer.model.train()
                    total_loss = 0.
                    for _ in range(self.trainer.gradient_accumulate_every):
                        data = next(self.trainer.dl).to(device)
                        with self.trainer.accelerator.autocast():
                            loss = self.trainer.model(data)
                            loss = loss / self.trainer.gradient_accumulate_every
                            total_loss += loss.item()
                        self.trainer.accelerator.backward(loss)
                    gpu_stats = self.memory_monitor.get_stats_string()
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.set_postfix_str(gpu_stats)
                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.trainer.model.parameters(), self.trainer.max_grad_norm)
                    self.trainer.opt.step()
                    self.trainer.opt.zero_grad()
                    accelerator.wait_for_everyone()
                    self.trainer.step += 1
                    if accelerator.is_main_process:
                        self.trainer.ema.update()
                        if self.trainer.step != 0 and self.trainer.step % self.trainer.save_and_sample_every == 0:
                            milestone = self.trainer.step // self.trainer.save_and_sample_every
                            accelerator.print(f"Saving checkpoint at step {self.trainer.step}...")
                            self.trainer.save(milestone)
                            self.trainer.ema.ema_model.eval()
                            with torch.inference_mode():
                                from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups
                                def set_flash_attn(model, enable):
                                    for module in model.modules():
                                        if hasattr(module, 'flash'):
                                            module.flash = enable
                                set_flash_attn(self.trainer.ema.ema_model, False)
                                try:
                                    accelerator.print("Generating samples...")
                                    batches = num_to_groups(self.trainer.num_samples, self.trainer.batch_size)
                                    all_images_list = list(map(lambda n: self.trainer.ema.ema_model.sample(batch_size=n), batches))
                                    accelerator.print("✓ Sample generation successful")
                                except Exception as e:
                                    accelerator.print(f"✗ Sampling failed: {e}")
                                    all_images_list = None
                                if all_images_list is not None:
                                    all_images = torch.cat(all_images_list, dim=0)
                                    from torchvision import utils
                                    import math
                                    utils.save_image(all_images, str(self.trainer.results_folder / f'sample-{milestone}.png'),
                                                   nrow=int(math.sqrt(self.trainer.num_samples)))
                                    accelerator.print(f"✓ Saved samples to sample-{milestone}.png")
                                    if self.trainer.save_best_and_latest_only:
                                        self.trainer.save("latest")
                    pbar.update(1)
            accelerator.print('training complete')
        self.trainer.train = patched_train

    def train(self):
        self.start_time = time.time()
        print(f"\n{'='*70}\n{'STARTING EXP 2: LOW LR + MILD AUG':^70}\n{'='*70}")
        print(f"  Training steps:   {self.trainer.train_num_steps:,}")
        print(f"  Learning rate:    {self.trainer.opt.param_groups[0]['lr']:.2e}")
        print(f"  Augmentation:     RandomHorizontalFlip")
        print(f"{'='*70}\n")
        self.memory_monitor.start_background_monitoring()
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            print("Training interrupted!")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            self.memory_monitor.stop_background_monitoring()

if __name__ == '__main__':
    device = get_device()
    print_gpu_info(device)
    memory_monitor = GPUMemoryMonitor(device)

    print("Loading model...")
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
    )

    import denoising_diffusion_pytorch.denoising_diffusion_pytorch as ddp_module
    original_cpu_count_fn = ddp_module.cpu_count
    ddp_module.cpu_count = lambda: 16

    trainer = Trainer(
        diffusion,
        folder='./sysu-shape-dataset/combined/',
        train_batch_size=64,
        train_lr=2e-5,          # EXPERIMENT 2: LOW LEARNING RATE
        train_num_steps=100000, # EXPERIMENT 2: MORE STEPS
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        save_and_sample_every=4000,
        results_folder='./results_exp2_low_lr_mild_aug', # EXPERIMENT 2 FOLDER
        num_samples=64,
        calculate_fid=False,
    )

    print("Replacing dataset with DatasetMildAug...")
    ds = DatasetMildAug('./sysu-shape-dataset/combined/', 64)
    dl = DataLoader(ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=16)
    dl = trainer.accelerator.prepare(dl)
    trainer.dl = cycle(dl)
    trainer.ds = ds
    ddp_module.cpu_count = original_cpu_count_fn

    monitored_trainer = MonitoredTrainer(trainer, memory_monitor)
    monitored_trainer.train()

