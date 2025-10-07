#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual download script for ReinFlow pre-trained model checkpoints
"""
import os
import gdown
from omegaconf import OmegaConf

# Import download function from project
from script.download_url import get_checkpoint_download_url

def download_checkpoint(env, model_type="diffusion", task_type="img"):
    """
    Manually download specified checkpoint
    
    Args:
        env: Environment name like 'square', 'lift', 'can'
        model_type: Model type like 'diffusion', 'reflow', 'shortcut'  
        task_type: Task type 'img' or 'state'
    """
    
    # Construct config path
    if model_type == "diffusion":
        if task_type == "img":
            base_path = f"log/robomimic/pretrain/{env}/{env}_pre_diffusion_mlp_img_ta4_td100/checkpoint/state_4000.pt"
        else:
            base_path = f"log/robomimic/pretrain/{env}/{env}_pre_diffusion_mlp_ta4_td20/checkpoint/state_5000.pt"
    elif model_type == "reflow":
        base_path = f"log/robomimic/pretrain/{env}/{env}/ReFlow/state_2000.pt"
    elif model_type == "shortcut":
        base_path = f"log/robomimic/pretrain/{env}/{env}/ShortCut/state_2000.pt"
    else:
        print(f"Unsupported model type: {model_type}")
        return

    # Create config object
    cfg = OmegaConf.create({
        'base_policy_path': base_path
    })
    
    try:
        # Get download URL
        download_url = get_checkpoint_download_url(cfg)

        if download_url is None:
            print(f"Download URL not found for {env} {model_type}")
            return

        # Create target directory
        target_dir = os.path.dirname(base_path)
        os.makedirs(target_dir, exist_ok=True)

        print(f"Downloading {env} {model_type} model...")
        print(f"From: {download_url}")
        print(f"To: {base_path}")

        # Download file
        gdown.download(url=download_url, output=base_path, fuzzy=True)
        print(f"✓ Download complete: {base_path}")

    except Exception as e:
        print(f"✗ Download failed: {e}")

if __name__ == "__main__":
    # Example: download different models
    print("Starting ReinFlow pre-trained model download...")

    # Download different models for Square environment
    download_checkpoint("square", "diffusion", "img")
    download_checkpoint("square", "reflow", "img")
    download_checkpoint("square", "shortcut", "img")

    print("\nAll downloads complete!")