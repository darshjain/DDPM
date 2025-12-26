# DDPM Implementation on SysU-Shape Dataset
**Course:** MLIA (Machine Learning in Image Analysis)  
**Project:** Denoising Diffusion Probabilistic Models (DDPM)

### Team Members
- **zjy6us**
- **bhj4jy**
- **wjw8yd**
- **dny9jg**

---

## 📂 Project Structure

```
├── train.py                  # Baseline training script (Dim 64, Mild Augmentation)
├── train_exp1_low_lr.py      # Experiment 1: Low Learning Rate
├── train_exp2_hybrid.py      # Experiment 2: Low LR + Mild Augmentation
├── train_exp3_deeper.py      # Experiment 3: Deeper Model (Dim 96)
├── train_exp4_dim128.py      # Experiment 4: Best Model (Dim 128)
│
├── run.sbatch                # SLURM script for baseline training
├── run_exp[1-4].sbatch       # SLURM scripts for experiments
│
├── evaluation_scripts/       # Scripts for FID/IS calculation and plotting
│   ├── eval_utils.py         # Shared evaluation logic
│   ├── plot_all_results.py   # Generates comparison plots
│   └── run_eval_*.sbatch     # Evaluation job scripts
│
├── plot_training_loss.py     # Script to generate loss curves from logs
├── gpu_monitor.py            # Utility for tracking GPU usage
└── requirements.txt          # Python dependencies
```

> **Note:** Pre-trained models and full result logs are stored on Rivanna shared storage at `/standard/mlia/[GROUP_FOLDER]` due to size constraints.

---

## 🚀 Usage Instructions

### 1. Setup Environment
Ensure you are on a GPU node (e.g., via `interactive` or `sbatch`).
```bash
module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

# Install dependencies
pip install -r requirements.txt
```

### 2. Training
To train the baseline model:
```bash
sbatch run.sbatch
```
To train the best-performing model (Experiment 4):
```bash
sbatch run_exp4.sbatch
```
Training takes approximately 8-12 hours on an A6000 GPU.

### 3. Evaluation
To compute FID and Inception Scores for a trained model:
```bash
cd evaluation_scripts
sbatch run_eval_exp4.sbatch
```
This generates `evaluation_results.json` in the respective results folder.

### 4. Visualization
To generate the comparison plots (FID/IS curves):
```bash
cd evaluation_scripts
python plot_all_results.py
```
Outputs will be saved as `comparison_fid.png` and `comparison_is.png`.

To generate loss curves:
```bash
python plot_training_loss.py
```
Outputs will be saved in `loss_plots/`.

---

## 📊 Experiments & Results

We conducted 5 variations to optimize generation quality:
1.  **Baseline**: Dim 64, LR 1e-4.
2.  **Low LR**: Reduced learning rate to 2e-5.
3.  **Low LR + Aug**: Added horizontal flip augmentation.
4.  **Dim 96**: Increased model capacity.
5.  **Dim 128 (Best)**: Further increased capacity.

**Best Result (Exp 4):**
- **FID:** ~59.01 (Lower is better)
- **IS:** ~6.57 (Higher is better)

Full analysis and graphs are provided in the accompanying report.
