"""
STLM runner: integrates STLM into the project's main.py dispatch.
Runs STLM training and evaluation for a given category using original paper settings.
"""

import argparse
import os
import torch
from stlm_code.trainSTLM import train
from utils.model_benchmark import (
    reset_gpu_peak, measure_gpu_memory_mb,
)


def run_stlm(category, mvtec_path="./data/MVTec-AD/", dtd_path="./data/dtd/images/",
             mobile_sam_path="./weights/mobile_sam.pt",
             sam_vit_h_path="./weights/sam_vit_h_4b8939.pth",
             gpu_id=0, num_workers=8, steps=200, eval_per_steps=5,
             backbone_key=None, backbone_bench=None):
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
        backbone_key: None/"tinyvit" for original TinyViT, or backbone key for custom encoder

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
    enc_name = backbone_key if backbone_key and backbone_key.lower() != "tinyvit" else "TinyViT"
    print(f"  teacher: SAM ViT-H ({sam_vit_h_path})")
    print(f"  student encoder: {enc_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        reset_gpu_peak(device)
        with torch.cuda.device(gpu_id):
            raw = train(args, category, rotate_90=rotate_90, random_rotate=random_rotate, backbone_key=backbone_key)
        gpu_train_mb = measure_gpu_memory_mb(device)
    else:
        print("  WARNING: No CUDA GPU found. Running on CPU (will be very slow).")
        raw = train(args, category, rotate_90=rotate_90, random_rotate=random_rotate, backbone_key=backbone_key)
        gpu_train_mb = 0.0

    # Normalize metric keys to match project's print_summary_table() format
    metrics = {
        "image_auroc": raw.get("auc_detect_fa", 0),
        "pixel_auroc": raw.get("auc_fa", 0),
        "aupro_0.3": raw.get("aupro_fa", 0),
        "backbone_benchmark": backbone_bench,
        "gpu_train_mb": round(gpu_train_mb, 1),
        "stlm_raw": raw,  # keep original STLM metrics for reference
    }
    return metrics
