#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gdown

def download_sample_checkpoint():
    """Download a sample checkpoint using direct Google Drive link"""
    
    # Example: Download Square ReFlow checkpoint
    url = "https://drive.google.com/file/d/1nC4yc9XjXO1YtZmFoWh1No5zn3yhtVeo/view?usp=drive_link"
    output_path = "log/robomimic/pretrain/square/ReFlow/state_2000.pt"
    
    # Create directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Downloading checkpoint to: {output_path}")
    try:
        gdown.download(url, output_path, fuzzy=True)
        print("Download successful!")
        print(f"File saved at: {output_path}")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_sample_checkpoint()