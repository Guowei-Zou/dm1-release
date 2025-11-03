#!/usr/bin/env python3
"""
Simple script to download DM1 pretrained weights from Google Drive
Google Drive folder: https://drive.google.com/drive/folders/1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8
"""

import os
import sys
import subprocess

# Google Drive folder ID
FOLDER_ID = "1l5JZvx9OBXRW0A6Vy27aO967J85DaKh8"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
OUTPUT_DIR = "dm1_pretraining_checkpoints"


def check_gdown():
    """Check if gdown is installed"""
    try:
        import gdown
        return True
    except ImportError:
        return False


def install_gdown():
    """Install gdown using pip"""
    print("Installing gdown...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("✓ gdown installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install gdown: {e}")
        return False


def download_folder():
    """Download the entire Google Drive folder"""
    import gdown

    print("=" * 60)
    print("DM1 Pretrained Checkpoints Downloader")
    print("=" * 60)
    print(f"Source: {FOLDER_URL}")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading checkpoints from Google Drive...")
    print("This may take several minutes depending on your connection...")
    print()

    try:
        # Download folder
        gdown.download_folder(
            url=FOLDER_URL,
            output=OUTPUT_DIR,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )

        print()
        print("=" * 60)
        print("✓ Download completed!")
        print("=" * 60)
        print()

        # Verify downloaded files
        verify_downloads()

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("⚠ Download failed or incomplete")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Alternative methods:")
        print("1. Manual download from: " + FOLDER_URL)
        print("2. Try: gdown --folder " + FOLDER_URL + " -O " + OUTPUT_DIR)
        print("3. Check if the folder is publicly accessible")
        print()

        return False


def verify_downloads():
    """Verify downloaded checkpoint files"""
    if not os.path.exists(OUTPUT_DIR):
        print(f"⚠ Output directory not found: {OUTPUT_DIR}")
        return

    # Count .pt files
    pt_files = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))

    print(f"Found {len(pt_files)} checkpoint files (.pt)")
    print()

    # Expected structure
    w_settings = ["w_0p1", "w_0p5", "w_0p9"]
    tasks = ["lift", "can", "square", "transport"]

    print("Checkpoint structure:")
    for w in w_settings:
        for task in tasks:
            task_dir = os.path.join(OUTPUT_DIR, w, task)
            if os.path.exists(task_dir):
                files = [f for f in os.listdir(task_dir) if f.endswith('.pt')]
                status = "✓" if len(files) >= 8 else "⚠"
                print(f"  {status} {w}/{task}: {len(files)} files")
            else:
                print(f"  ✗ {w}/{task}: Not found")

    print()
    print(f"Checkpoints saved in: {os.path.abspath(OUTPUT_DIR)}")


def main():
    print()

    # Check and install gdown if needed
    if not check_gdown():
        print("gdown is not installed.")
        response = input("Install gdown now? (y/n): ").strip().lower()
        if response == 'y':
            if not install_gdown():
                print("Failed to install gdown. Please install manually:")
                print("  pip install gdown")
                sys.exit(1)
        else:
            print("gdown is required. Install with: pip install gdown")
            sys.exit(1)

    # Download folder
    success = download_folder()

    if success:
        print()
        print("=" * 60)
        print("Download completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Check the downloaded checkpoints in:", OUTPUT_DIR)
        print("2. Use them in your experiments by setting checkpoint paths")
        print()
    else:
        print()
        print("Please try manual download or alternative methods.")
        sys.exit(1)


if __name__ == "__main__":
    main()
