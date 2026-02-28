"""
STLM training script.
Ported from https://github.com/Qi5Lei/STLM with minimal changes.
Adjusted paths to work within this project structure.
"""

import argparse
import os
import warnings
import torch

# Reduce CUDA memory fragmentation (helps when reserved-but-unallocated memory is large)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stlm_code.constant import RESIZE_SHAPE, ALL_CATEGORY
from stlm_code.data.mvtec_dataset import MVTecDataset
from stlm_code.evalSTLM import evaluate
from stlm_code.model.losses import cosine_similarity_loss, focal_loss, l1_loss
from stlm_code.mob_sam import Batch_Sam, Batch_SamE, SegmentationNet
from stlm_code.model.model_utils import l2_norm

warnings.filterwarnings("ignore")


def train(args, category, rotate_90=False, random_rotate=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Initialize student (two-stream TinyViT)
    print("  Loading student model (MobileSAM TinyViT)...")
    sam_checkpoint = args.mobile_sam_path
    model_type = "vit_t"
    sam_mode = "train"
    twostream = Batch_SamE(sam_checkpoint, model_type, sam_mode, device)
    print("  Student model loaded.")

    # Initialize teacher (SAM ViT-H, frozen)
    print("  Loading teacher model (SAM ViT-H, 2.4GB)... This may take a while.")
    sam_checkpoint = args.sam_vit_h_path
    model_type = "vit_h"
    sam_mode = "eval"
    fix_teacher = Batch_Sam(sam_checkpoint, model_type, sam_mode, device)
    print("  Teacher model loaded.")

    # Feature aggregation module
    segmentation_net = SegmentationNet(512).to(device)
    print("  All models loaded. Starting training...")

    # Define optimizers
    tlm_optimizer = torch.optim.Adam(twostream.parameters(), betas=(0.5, 0.999), lr=0.0005)
    seg_optimizer = torch.optim.SGD(
        [
            {"params": segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0
    flag = True
    import time

    best_metrics = {"aupro_fa": 0, "auc_fa": 0, "auc_detect_fa": 0}

    while flag:
        epoch_start = time.time()
        num_batches = len(dataloader)
        for batch_idx, data in enumerate(dataloader):
            print(f"  Epoch {global_step+1}/{args.steps} | Batch {batch_idx+1}/{num_batches}", end="\r", flush=True)
            twostream.train()
            segmentation_net.train()
            tlm_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            img_origin = data["img_origin"].to(device)
            img_pseudo = data["img_aug"].to(device)
            mask = data["mask"].to(device)

            pfeature1, pfeature2 = fix_teacher(img_pseudo)
            dfeature1, dfeature2 = fix_teacher(img_origin)
            Tpfeature = [pfeature1, pfeature2]
            Tdfeature = [dfeature1, dfeature2]
            Pfeature, Dfeature = twostream(img_pseudo)

            outputs_Tplain = [
                l2_norm(output_p.detach()) for output_p in Tpfeature
            ]
            outputs_Tdenoising = [
                l2_norm(output_d.detach()) for output_d in Tdfeature
            ]

            outputs_Splain = [
                l2_norm(output_p) for output_p in Pfeature
            ]
            outputs_Sdenoising = [
                l2_norm(output_d) for output_d in Dfeature
            ]

            output_pain_list = []
            for output_t, output_s in zip(outputs_Tplain, outputs_Splain):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_pain_list.append(a_map)

            output_denoising_list = []
            for output_t, output_s in zip(outputs_Tdenoising, outputs_Sdenoising):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_denoising_list.append(a_map)

            output = torch.cat(
                [
                    F.interpolate(
                        -output_p * output_d,
                        size=outputs_Sdenoising[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_p, output_d in zip(outputs_Splain, outputs_Sdenoising)
                ],
                dim=1,
            )

            output_segmentation = segmentation_net(output)

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cos_loss = cosine_similarity_loss(output_pain_list) + cosine_similarity_loss(output_denoising_list)

            f_loss = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_l = l1_loss(output_segmentation, mask)
            seg_loss = f_loss + l1_l

            total_loss_val = seg_loss + cos_loss
            total_loss_val.backward()
            tlm_optimizer.step()
            seg_optimizer.step()

        global_step += 1
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {global_step}/{args.steps} completed in {epoch_time:.1f}s | loss={total_loss_val.item():.4f}")

        if global_step % args.eval_per_steps == 0:
            aupro_tlm, auc_tlm, auc_detect_tlm, aupro_fa, auc_fa, auc_detect_fa = evaluate(
                args, category, twostream, segmentation_net
            )
            print(
                f"[{category}] Epoch {global_step}/{args.steps} | "
                f"TLM: AUPRO={aupro_tlm:.4f} P-AUC={auc_tlm:.4f} I-AUC={auc_detect_tlm:.4f} | "
                f"FA:  AUPRO={aupro_fa:.4f} P-AUC={auc_fa:.4f} I-AUC={auc_detect_fa:.4f}"
            )

            if auc_fa > best_metrics["auc_fa"]:
                best_metrics = {
                    "aupro_fa": aupro_fa,
                    "auc_fa": auc_fa,
                    "auc_detect_fa": auc_detect_fa,
                    "aupro_tlm": aupro_tlm,
                    "auc_tlm": auc_tlm,
                    "auc_detect_tlm": auc_detect_tlm,
                    "epoch": global_step,
                }

        if global_step >= args.steps:
            flag = False
            break

    print(f"[{category}] Best FA results @ epoch {best_metrics.get('epoch', 'N/A')}: {best_metrics}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="STLM Training")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)

    # Paths - adjusted for this project
    parser.add_argument("--mvtec_path", type=str, default="./data/MVTec-AD/")
    parser.add_argument("--dtd_path", type=str, default="./data/dtd/images/")
    parser.add_argument("--mobile_sam_path", type=str, default="./weights/mobile_sam.pt")
    parser.add_argument("--sam_vit_h_path", type=str, default="./weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")

    # Hyperparameters (paper defaults)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_per_steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--T", type=int, default=100)

    # Category selection
    parser.add_argument("--category", type=str, default=None,
                        help="Train single category. If None, train all.")

    args = parser.parse_args()

    # Category rotation settings from paper
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

    with torch.cuda.device(args.gpu_id):
        if args.category:
            # Single category mode
            cat = args.category
            rotate_90 = cat in rotation_category
            random_rotate = 5 if (cat in slight_rotation_category or cat in rotation_category) else 0
            print(f"Training {cat} (rotate_90={rotate_90}, random_rotate={random_rotate})")
            train(args, cat, rotate_90=rotate_90, random_rotate=random_rotate)
        else:
            # All categories (original paper behavior)
            all_results = {}
            for obj in no_rotation_category:
                print(f"\n=== {obj} (no rotation) ===")
                all_results[obj] = train(args, obj)

            for obj in slight_rotation_category:
                if obj not in all_results:
                    print(f"\n=== {obj} (slight rotation) ===")
                    all_results[obj] = train(args, obj, rotate_90=False, random_rotate=5)

            for obj in rotation_category:
                if obj not in all_results:
                    print(f"\n=== {obj} (rotation) ===")
                    all_results[obj] = train(args, obj, rotate_90=True, random_rotate=5)

            print("\n=== Final Results ===")
            for cat, metrics in all_results.items():
                print(f"{cat}: {metrics}")


if __name__ == "__main__":
    main()
