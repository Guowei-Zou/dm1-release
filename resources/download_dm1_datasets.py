#!/usr/bin/env python3
"""Selective downloader for DM1 pre-training datasets.

By default only the Robomimic image datasets required for DM1 evaluation are
retrieved. Optional command line flags allow downloading additional dataset
suites (Gym, Kitchen, D3IL) when needed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Mapping from dataset group to the individual Google Drive folders that store
# the processed train/normalization files expected by the configs.
DATASET_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "robomimic": [
        {
            "name": "Robomimic Lift (image)",
            "url": "https://drive.google.com/drive/u/1/folders/1H-UncdzHx6wd5NWVzrQyftfGls7KGz1O",
            "output": "robomimic/lift-img",
        },
        {
            "name": "Robomimic Can (image)",
            "url": "https://drive.google.com/drive/u/1/folders/1VGp_5xXXb1-GJutdSc6AZSzXNk-6_vRz",
            "output": "robomimic/can-img",
        },
        {
            "name": "Robomimic Square (image)",
            "url": "https://drive.google.com/drive/u/1/folders/1-aGqVeKLIzJCEst8p0ZTjfjkrXFfJLxa",
            "output": "robomimic/square-img",
        },
        {
            "name": "Robomimic Transport (image)",
            "url": "https://drive.google.com/drive/u/1/folders/1cOkAZQmmETYEPFrnnX0EuD6mv0kUfMO2",
            "output": "robomimic/transport-img",
        },
    ],
    "gym": [
        {
            "name": "Gym D4RL bundle (hopper, walker)",
            "url": "https://drive.google.com/drive/folders/13QiGNv3-RE9DdmAZ7HvKKb0KRrmGUoyg",
            "output": "gym",
        },
        {
            "name": "Gym Ant medium-expert (D4RL)",
            "url": "https://drive.google.com/drive/folders/1dZHv_DxEN3Yukfw5B8LtRQK7nnbmwzKE",
            "output": "gym/ant-medium-expert-v2",
        },
        {
            "name": "Gym Humanoid medium v3", 
            "url": "https://drive.google.com/drive/folders/1J6nDPwiNRoecn1M8aEagyTKOBhaVJ0Mo",
            "output": "gym/humanoid-medium-v3",
        },
    ],
    "kitchen": [
        {
            "name": "Kitchen complete", 
            "url": "https://drive.google.com/drive/u/1/folders/18aqg7KIv-YNXohTsRR7Zmg-RyDtdhkLc",
            "output": "gym/kitchen-complete-v0",
        },
        {
            "name": "Kitchen partial",
            "url": "https://drive.google.com/drive/u/1/folders/1zLOx1q4FbJK1ZWLui_vhM2x1fMEkBC2D",
            "output": "gym/kitchen-partial-v0",
        },
        {
            "name": "Kitchen mixed",
            "url": "https://drive.google.com/drive/u/1/folders/1HRMM16UC10A00oBqjYOL1E8hS5icwtvo",
            "output": "gym/kitchen-mixed-v0",
        },
    ],
    "d3il": [
        {
            "name": "D3IL Avoid (mode d56_r12)",
            "url": "https://drive.google.com/drive/u/1/folders/1ZAPvLQwv2y4Q98UDVKXFT4fvGF5yhD_o",
            "output": "d3il/avoid_m1",
        },
        {
            "name": "D3IL Avoid (mode d57_r12)",
            "url": "https://drive.google.com/drive/u/1/folders/1wyJi1Zbnd6JNy4WGszHBH40A0bbl-vkd",
            "output": "d3il/avoid_m2",
        },
        {
            "name": "D3IL Avoid (mode d58_r12)",
            "url": "https://drive.google.com/drive/u/1/folders/1mNXCIPnCO_FDBlEj95InA9eWJM2XcEEj",
            "output": "d3il/avoid_m3",
        },
    ],
}

DEFAULT_GROUPS: List[str] = ["robomimic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DM1 pre-training datasets with selectable suites."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(list(DATASET_GROUPS.keys()) + ["all"]),
        default=None,
        help="Dataset groups to download. Defaults to robomimic only.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root directory where datasets will be stored (default: data).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-download summary step.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available groups and exit without downloading.",
    )
    return parser.parse_args()


def check_gdown() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def install_gdown() -> bool:
    print("gdown is not installed. Attempting to install via pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("[OK] gdown installed successfully.")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to install gdown: {exc}")
        return False


def iter_selected_groups(selected: List[str]) -> List[str]:
    if not selected:
        return DEFAULT_GROUPS
    if "all" in selected:
        return sorted(DATASET_GROUPS.keys())
    return selected


def download_group(group: str, output_root: Path) -> None:
    import gdown

    entries = DATASET_GROUPS[group]
    print("=" * 60)
    print(f"Downloading group: {group}")
    print("=" * 60)

    for entry in entries:
        target_dir = output_root / entry["output"]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"-> {entry['name']}")
        print(f"   URL : {entry['url']}")
        print(f"   Dest: {target_dir}")
        gdown.download_folder(
            url=entry["url"],
            output=str(target_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )


def verify_downloads(output_root: Path) -> None:
    if not output_root.exists():
        print(f"[WARN] Directory not found: {output_root}")
        return

    npz_files: List[Path] = [
        path
        for path in output_root.rglob("*.npz")
        if path.is_file()
    ]
    print()
    print("Summary")
    print("-" * 60)
    print(f"Found {len(npz_files)} .npz files under {output_root.resolve()}")
    top_level = sorted([p for p in output_root.iterdir() if p.is_dir()])
    if top_level:
        print("Top-level directories:")
        for item in top_level:
            print(f"  - {item.relative_to(output_root)}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    if args.list:
        print("Available dataset groups:")
        for key in sorted(DATASET_GROUPS.keys()):
            items = DATASET_GROUPS[key]
            names = ", ".join(entry["name"] for entry in items)
            print(f"  - {key}: {names}")
        return

    if not check_gdown():
        response = input("gdown is required. Install it now? (y/n): ").strip().lower()
        if response == "y":
            if not install_gdown():
                print("Install gdown manually with: pip install gdown")
                sys.exit(1)
        else:
            print("Aborting download because gdown is missing.")
            sys.exit(1)

    groups = iter_selected_groups(args.groups)
    print(f"Selected groups: {', '.join(groups)}")
    output_root.mkdir(parents=True, exist_ok=True)

    for group in groups:
        download_group(group, output_root)

    if not args.skip_verify:
        verify_downloads(output_root)


if __name__ == "__main__":
    main()
