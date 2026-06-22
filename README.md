# Active few-shot segmentation by reinforcing data selection

Abstract:



---

## Project Structure

```
reptile2/
├── data/               ← put your raw NIfTI dataset here (see below)
├── data-resize/        ← auto-generated after running main.py
├── main.py             ← entry point, runs the full pipeline
├── reptile.py          ← 3D U-Net model + Reptile training logic
├── RL.py               ← PPO-based active support selection
├── data_loader.py      ← NIfTI data loading and batch generation
├── resize.py           ← resizes volumes to a fixed spatial size
├── split.py            ← train/val/test split utilities
└── requirements.txt
```

---

## Setup & How to Run

### 1. Place the dataset

Put your NIfTI dataset inside the `./data` folder. The data loader expects files named like:

```
data/
├── case_001_img.nii
├── case_001_mask.nii
├── case_002_img.nii
├── case_002_mask.nii
└── ...
```

Each image file must have a corresponding mask file with the same prefix.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> A GPU is recommended (CUDA). The code will fall back to CPU automatically if no GPU is available, but training will be much slower.

### 3. Run the pipeline

```bash
python3 main.py
```

This runs the full pipeline end-to-end:

| Step | What it does |
|------|-------------|
| 1 | Resizes all volumes to `(256, 256, 32)` and saves them to `./data-resize` |
| 2 | Splits cases into train / val / test and writes `train.txt`, `val.txt`, `test.txt` |
| 3 | Trains the Reptile meta-learning initialisation |
| 4 | Saves the trained weights to `reptile_init.pt` |
| 5 | Runs the PPO RL agent for active support selection |

---

## Hyperparametrs settings

The following table summarzies the hyperparameters used in the final run:

#### Meta-Learning (Reptile) Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| REPTILE_BATCH  | 8     | Batch size for Reptile training |
| N_SUPPORT      | 4     | Number of support samples per task |
| OUTER_STEPS    | 10,000   | Number of meta-training iterations |
| INNER_STEPS    | 4     | Number of adaptation steps|
| INNER_LR       | 0.0001 | Inner-loop learning rate |
| OUTER_LR       | 0.001   | Decaying Meta-learning rate |
| INNER_Loop_Optimizer |Stochastic Gradient Descent|

#### Reinforcement Learning Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| RL_BATCH       | 64    | RL batch size per episode |
| RL_TRAIN_TRIALS| 1M    | Number of training trials |
| RL_STEPS       | 256   | Training steps per trial |

## Configuration

All hyperparameters are defined at the top of `main.py` and can be changed there, for example a training configuration may look like:

```python
RESIZE_DIMS     = (256, 256, 32)   # target volume size
TRAIN_RATIO     = 0.7
OUTER_STEPS     = 100              # Reptile outer loop iterations
INNER_STEPS     = 5                # inner gradient steps per task
INNER_LR        = 1e-3
OUTER_LR        = 0.1
TASK_NUMBER     = 1                # which anatomical class to segment
RL_TRAIN_TRIALS = 10               # number of PPO training trials
```

---

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- nibabel
- scikit-image
- numpy, scipy, tqdm
