#!/bin/bash
# Quick download script for ReinFlow checkpoints

echo "ReinFlow Checkpoint Downloader"
echo "============================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p log/robomimic/pretrain/square/ReFlow/
mkdir -p log/robomimic/pretrain/lift/
mkdir -p log/robomimic/pretrain/can/

echo "Downloading checkpoints using gdown..."

# Download Square ReFlow checkpoint (example)
echo "1. Downloading Square ReFlow checkpoint..."
python3 -c "
import gdown
import os
url = 'https://drive.google.com/file/d/1nC4yc9XjXO1YtZmFoWh1No5zn3yhtVeo/view?usp=drive_link'
output = 'log/robomimic/pretrain/square/ReFlow/state_2000.pt'
os.makedirs(os.path.dirname(output), exist_ok=True)
try:
    gdown.download(url, output, fuzzy=True)
    print('✓ Downloaded: ' + output)
except Exception as e:
    print('✗ Failed: ' + str(e))
"

echo "Download script completed!"
echo "Check the log/ directory for downloaded files."