#!/bin/bash

# DM1 Environment Activation Script
# Usage: source activate_dm1.sh

echo "Activating DM1 environment..."

# Set DM1-specific environment variables (must be set before imports)
# UPDATE THESE PATHS to match your installation directory
export DM1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DM1_DATA_DIR="$DM1_DIR/data"
export DM1_LOG_DIR="$DM1_DIR/log"
export DM1_WANDB_ENTITY="your-wandb-entity"  # Update this with your wandb entity
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# For backward compatibility with ReinFlow code
export REINFLOW_DIR="$DM1_DIR"
export REINFLOW_DATA_DIR="$DM1_DATA_DIR"
export REINFLOW_LOG_DIR="$DM1_LOG_DIR"
export REINFLOW_WANDB_ENTITY="$DM1_WANDB_ENTITY"

# Activate conda environment
# UPDATE THIS PATH to match your conda installation
CONDA_BASE="${CONDA_PREFIX:-${HOME}/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "Warning: conda.sh not found. Please activate conda environment manually."
fi

conda activate dm1

# Load environment variables
source ~/.bashrc

# Change to DM1 directory
cd "$DM1_DIR"

echo "DM1 environment activated successfully!"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "DM1_DIR: $DM1_DIR"

# Test core imports
python -c "
try:
    import mujoco_py
    import d4rl
    import robosuite
    import torch
    import hydra
    print('✅ All core dependencies imported successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

