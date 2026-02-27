"""
STLM runner: integrates STLM into the project's main.py dispatch.
Runs STLM training and evaluation for a given category using original paper settings.
"""

import argparse
import os
import torch
from stlm_code.trainSTLM import train


def run_stlm(category, mvtec_path="./data/MVTec-AD/", dtd_path="./data/dtd/images/",
             mobile_sam_path="./weights/mobile_sam.pt",
             sam_vit_h_path="./weights/sam_vit_h_4b8939.pth",
             gpu_id=0, num_workers=8, steps=200, eval_per_steps=5):
    """Run STLM training and evaluation for a single category.

    Args:
        category: MVTec AD category name (e.g., "bottle", "cable")
        mvtec_path: Path to MVTec AD dataset root
        dtd_path: Path to DTD texture images for pseudo-anomaly generation
        mobile_sam_path: Path to MobileSAM weights
        sam_vit_h_path: Path to SAM ViT-H weights
        gpu_id: GPU device ID
        num_workers: DataLoader workers
        steps: Training epochs
        eval_per_steps: Evaluate every N epochs

    Returns:
        dict with best metrics
    """

    # Category rotation settings from the paper
    no_rotation_category = [
        "capsule", "metal_nut", "pill", "toothbrush",
        "transistor", "screw", "grid",
    ]
    slight_rotation_category = [
        "wood", "zipper", "cable", "transistor", "screw", "grid",
    ]
    rotation_category = [
        "bottle", "grid", "hazelnut", "leather",
        "tile", "carpet", "screw", "transistor",
    ]

    # Determine rotation settings for this category
    rotate_90 = category in rotation_category
    random_rotate = 5 if (category in slight_rotation_category or category in rotation_category) else 0

    # Build args namespace to match STLM's expected interface
    args = argparse.Namespace(
        gpu_id=gpu_id,
        num_workers=num_workers,
        mvtec_path=mvtec_path if mvtec_path.endswith("/") else mvtec_path + "/",
        dtd_path=dtd_path,
        mobile_sam_path=mobile_sam_path,
        sam_vit_h_path=sam_vit_h_path,
        checkpoint_path="./saved_model/",
        bs=16,
        lr_res=0.1,
        lr_seghead=0.01,
        steps=steps,
        eval_per_steps=eval_per_steps,
        gamma=4,
        T=100,
    )

    print(f"\n=== STLM Training: {category} ===")
    print(f"  rotate_90={rotate_90}, random_rotate={random_rotate}")
    print(f"  steps={steps}, eval_per_steps={eval_per_steps}")
    print(f"  teacher: SAM ViT-H ({sam_vit_h_path})")
    print(f"  student: MobileSAM TinyViT ({mobile_sam_path})")

    if torch.cuda.is_available():
        with torch.cuda.device(gpu_id):
            metrics = train(args, category, rotate_90=rotate_90, random_rotate=random_rotate)
    else:
        print("  WARNING: No CUDA GPU found. Running on CPU (will be very slow).")
        metrics = train(args, category, rotate_90=rotate_90, random_rotate=random_rotate)

    return metrics
