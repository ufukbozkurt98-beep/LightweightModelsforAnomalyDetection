"""
STLM runner: integrates STLM into the project's main.py dispatch.
Runs STLM training and evaluation for a given category using original paper settings.
"""

import argparse
import gc
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from stlm_code.trainSTLM import train
from stlm_code.constant import RESIZE_SHAPE
from stlm_code.data.mvtec_dataset import MVTecDataset
from stlm_code.model.model_utils import l2_norm
from configs.config import REPORTS_DIR
from utils.model_benchmark import (
    reset_gpu_peak, measure_gpu_memory_mb,
)


def _measure_stlm_inference(args, category, twostream, segmentation_net, device):
    """Run a timed inference pass over the test set (student + SegNet only, no teacher).

    This measures pure inference latency consistently with CFlow/FastFlow/GLASS.
    Only the student forward pass and SegmentationNet are timed — metric
    computation (AUROC, AUPRO) is excluded.
    """
    twostream.eval()
    segmentation_net.eval()

    dataset = MVTecDataset(
        is_train=False,
        mvtec_dir=args.mvtec_path + category + "/test/",
        resize_shape=RESIZE_SHAPE,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
    )

    num_images = len(dataset)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warmup
        for sample in dataloader:
            img = sample["img"].to(device)
            pfeature, dfeature = twostream(img)
            outputs_plain = [l2_norm(p.detach()) for p in pfeature]
            outputs_denoising = [l2_norm(d.detach()) for d in dfeature]
            output = torch.cat(
                [F.interpolate(-op * od, size=outputs_denoising[0].size()[2:],
                               mode="bilinear", align_corners=False)
                 for op, od in zip(outputs_plain, outputs_denoising)],
                dim=1,
            )
            segmentation_net(output)
            break  # single batch warmup

        if use_cuda:
            torch.cuda.synchronize()

        # Timed pass
        t0 = time.perf_counter()
        for sample in dataloader:
            img = sample["img"].to(device)
            pfeature, dfeature = twostream(img)
            outputs_plain = [l2_norm(p.detach()) for p in pfeature]
            outputs_denoising = [l2_norm(d.detach()) for d in dfeature]
            output = torch.cat(
                [F.interpolate(-op * od, size=outputs_denoising[0].size()[2:],
                               mode="bilinear", align_corners=False)
                 for op, od in zip(outputs_plain, outputs_denoising)],
                dim=1,
            )
            segmentation_net(output)

        if use_cuda:
            torch.cuda.synchronize()
        total_s = time.perf_counter() - t0

    per_image_ms = (total_s / num_images) * 1000.0
    return {
        "total_time_s": round(total_s, 3),
        "num_images": num_images,
        "per_image_ms": round(per_image_ms, 2),
        "throughput_fps": round(1000.0 / per_image_ms, 1) if per_image_ms > 0 else 0,
    }


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
        backbone_bench: Pre-computed backbone benchmark results (from main.py)

    Returns:
        dict with best metrics and inference benchmark
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
            raw, twostream, segmentation_net = train(
                args, category, rotate_90=rotate_90,
                random_rotate=random_rotate, backbone_key=backbone_key
            )
        gpu_train_mb = measure_gpu_memory_mb(device)
    else:
        print("  WARNING: No CUDA GPU found. Running on CPU (will be very slow).")
        raw, twostream, segmentation_net = train(
            args, category, rotate_90=rotate_90,
            random_rotate=random_rotate, backbone_key=backbone_key
        )
        gpu_train_mb = 0.0

    # Free teacher memory before inference measurement (teacher is a local in train(),
    # now out of scope, but CUDA cache may still hold it)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Inference benchmark (student + SegNet only, no teacher needed)
    print(f"  Measuring inference latency on test set...")
    reset_gpu_peak(device)
    inference_bench = _measure_stlm_inference(args, category, twostream, segmentation_net, device)
    gpu_infer_mb = measure_gpu_memory_mb(device)

    # Print per-category benchmark summary
    print(f"\n{'='*55}")
    print(f"  PER-CATEGORY BENCHMARK: {category.upper()}")
    print(f"{'='*55}")
    print(f"  Best epoch   : {raw.get('epoch', 'N/A')}")
    print(f"  I-AUROC (FA) : {raw.get('auc_detect_fa', 0) * 100:.2f}%")
    print(f"  P-AUROC (FA) : {raw.get('auc_fa', 0) * 100:.2f}%")
    print(f"  AUPRO (FA)   : {raw.get('aupro_fa', 0) * 100:.2f}%")
    print(f"  I-AUROC (TLM): {raw.get('auc_detect_tlm', 0) * 100:.2f}%")
    print(f"  P-AUROC (TLM): {raw.get('auc_tlm', 0) * 100:.2f}%")
    print(f"  AUPRO (TLM)  : {raw.get('aupro_tlm', 0) * 100:.2f}%")
    print(f"  GPU (train)  : {gpu_train_mb:.0f} MB")
    print(f"  GPU (infer)  : {gpu_infer_mb:.0f} MB")
    print(f"  Infer total  : {inference_bench['total_time_s']:.3f} s")
    print(f"  Infer/image  : {inference_bench['per_image_ms']:.2f} ms")
    print(f"  Throughput   : {inference_bench['throughput_fps']:.1f} FPS")
    print(f"{'='*55}\n")

    # Free teacher/training memory — only student + segnet needed from here
    del twostream, segmentation_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert STLM raw metrics from CUDA tensors to plain Python floats
    def _to_float(v):
        if isinstance(v, torch.Tensor):
            return round(float(v.cpu()), 4)
        return v
    raw_serializable = {k: _to_float(v) for k, v in raw.items()}

    # Normalize metric keys to match project's print_summary_table() format
    metrics = {
        "image_auroc": _to_float(raw.get("auc_detect_fa", 0)),
        "pixel_auroc": _to_float(raw.get("auc_fa", 0)),
        "aupro_0.3": _to_float(raw.get("aupro_fa", 0)),
        "inference_benchmark": inference_bench,
        "backbone_benchmark": backbone_bench,
        "gpu_train_mb": round(gpu_train_mb, 1),
        "gpu_infer_mb": round(gpu_infer_mb, 1),
        "stlm_raw": raw_serializable,  # keep original STLM metrics for reference
    }

    # Save per-category results as JSON
    enc_label = backbone_key if backbone_key and backbone_key.lower() != "tinyvit" else "tinyvit"
    results_dir = REPORTS_DIR / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{category}_stlm_{enc_label}_results.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  [saved → {out_path}]")

    return metrics
