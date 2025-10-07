# DM1 Quick Start Guide

**Complete guide from installation to training and evaluation.**

---

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 100GB+ disk space

---

## 🔧 Installation

### Step 1: Create Environment

```bash
conda create -n dm1 python=3.8 -y
conda activate dm1
```

### Step 2: Install MuJoCo

```bash
mkdir $HOME/.mujoco
cd ~/.mujoco

# Download mujoco210 (or use your existing installation)
# Add to ~/.bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Install mujoco_py
pip install 'cython<3.0.0'
# Follow detailed instructions in installation/reinflow-setup.md
```

### Step 3: Install Dependencies

```bash
cd ~/dm1-release

# For Franka Kitchen
pip install d4rl dm_control==1.0.16 mujoco==3.1.6

# For RoboMimic
pip install robomimic==0.3.0 robosuite==1.4.1

# Visualization (optional)
pip install wandb tqdm pandas seaborn
```

### Step 4: Setup Paths

```bash
bash ./script/set_path.sh
source ~/.bashrc
```

### Step 5: Install DM1

```bash
pip install -e .
```

**For detailed installation**, see [installation/reinflow-setup.md](installation/reinflow-setup.md)

---

## 🚀 Training

### Basic Training

```bash
conda activate dm1

# Train MeanFlow on Lift
python script/run.py agent=pretrain/train_meanflow_agent env=robomimic task=lift

# Train on Kitchen
python script/run.py agent=pretrain/train_meanflow_agent env=gym task=kitchen-partial-v0
```

### Other Models

```bash
# Reflow
python script/run.py agent=pretrain/train_reflow_agent env=robomimic task=lift

# Shortcut
python script/run.py agent=pretrain/train_shortcut_agent env=robomimic task=lift

# Diffusion
python script/run.py agent=pretrain/train_diffusion_agent env=robomimic task=lift
```

### Customize Parameters

```bash
python script/run.py agent=pretrain/train_meanflow_agent env=robomimic task=lift \
    seed=42 \
    model.learning_rate=1e-4 \
    training.batch_size=256 \
    training.num_epochs=2000 \
    model.dispersive_reg_coeff=0.1
```

### Key Hyperparameters

- `model.learning_rate`: Learning rate (default: 1e-4)
- `training.batch_size`: Batch size (default: 256)
- `training.num_epochs`: Training epochs (default: 2000)
- `model.dispersive_reg_coeff`: Dispersive regularization strength (default: 0.1)
- `seed`: Random seed

Check config files in `cfg/` for more hyperparameters.

---

## 📊 Evaluation

### Evaluate Trained Model

```bash
# Evaluate MeanFlow
python script/run.py agent=eval/eval_meanflow_agent env=robomimic task=lift \
    checkpoint_path=/path/to/model.pt

# Evaluate Reflow
python script/run.py agent=eval/eval_reflow_agent env=robomimic task=lift \
    checkpoint_path=/path/to/model.pt
```

### Monitor Training

Checkpoints are saved to: `log/<env>/<task>/<agent_name>/<timestamp>/`

If you configured WandB:
```bash
# Logs will be uploaded to your WandB account
# Project: dm1
# Entity: your-configured-entity
```
