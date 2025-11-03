#!/bin/bash
# Quick download script for DM1 pretrained checkpoints

echo "================================================"
echo "DM1 Pretrained Checkpoints Downloader"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Google Drive folder ID for dm1_pretraining_checkpoints
FOLDER_ID="1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8"
OUTPUT_DIR="dm1_pretraining_checkpoints"

echo "Google Drive folder: $FOLDER_ID"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "⚠ gdown is not installed. Installing..."
    pip install gdown
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Downloading checkpoints from Google Drive..."
echo "This may take several minutes depending on your connection..."
echo ""

# Method 1: Try folder download (recommended)
echo "Attempting folder download..."
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$OUTPUT_DIR" --remaining-ok

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download completed successfully!"
    echo ""
    echo "Verifying downloaded files..."

    # Count downloaded .pt files
    PT_COUNT=$(find "$OUTPUT_DIR" -name "*.pt" 2>/dev/null | wc -l)
    echo "Found $PT_COUNT checkpoint files (.pt)"

    # Show directory structure
    echo ""
    echo "Directory structure:"
    tree -L 3 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type d | head -20

else
    echo ""
    echo "⚠ Folder download failed or incomplete."
    echo ""
    echo "Alternative download methods:"
    echo "1. Manual download: https://drive.google.com/drive/folders/$FOLDER_ID"
    echo "2. Use Python script: python download_dm1_checkpoints.py"
    echo "3. Try with cookies: gdown --folder <URL> -O $OUTPUT_DIR --remaining-ok --fuzzy"
fi

echo ""
echo "================================================"
echo "Download process finished!"
echo "================================================"
echo ""
echo "Checkpoint location: $OUTPUT_DIR"
echo ""
