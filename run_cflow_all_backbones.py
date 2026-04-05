"""
Run CFlow-AD across multiple backbones, all 15 MVTec categories each.
Results are saved per backbone and resume-safe (skips completed categories).
"""
import json
import torch
from pathlib import Path

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad
from utils.feature_extractor import build_extractor
from utils.model_benchmark import run_all_benchmarks, print_benchmark_results
from runners.cflow_runner import run_cflow

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, TAR_PATH, VAL_RATIO_CFLOW,
)

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

BACKBONES = [
    "mobilenetv3_large",
    # "efficientnet_lite1",
    # "mobilevit_s",
    # "shufflenet_g3",
    # "mobileformer_508m",
]

CFLOW_OUT_INDICES = (2, 3, 4)


def get_channel_cap(backbone_key):
    """Auto-detect channel cap for wide-channel backbones."""
    tmp = build_extractor(backbone_key, pretrained=False)
    max_ch = max(tmp.feature_channels.values())
    del tmp
    if max_ch > 512:
        print(f"  Auto channel_cap=256 (backbone max channel={max_ch} > 512)")
        return 256
    return None


def print_summary_table(all_results, backbone):
    print(f"\n{'='*100}")
    print(f"  SUMMARY: CFLOW + {backbone}")
    print(f"{'='*100}")
    print(f"  {'Category':<14} {'Img AUROC':>10} {'Pix AUROC':>10} {'AUPRO':>10} "
          f"{'ms/img':>10} {'FPS':>10} {'Model FPS':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    img_sum, pix_sum, pro_sum = 0.0, 0.0, 0.0
    count = 0

    for cat in ALL_CATEGORIES:
        m = all_results.get(cat)
        if m is None:
            print(f"  {cat:<14} {'N/A':>10}")
            continue
        img = m["image_auroc"] * 100
        pix = m["pixel_auroc"] * 100
        pro = m["aupro_0.3"] * 100
        ms = m.get("inference_benchmark", {}).get("per_image_ms", 0)
        fps = m.get("inference_benchmark", {}).get("throughput_fps", 0)
        model_fps = m.get("fps_benchmark", {}).get("fps_all", 0)
        print(f"  {cat:<14} {img:>9.2f}% {pix:>9.2f}% {pro:>9.2f}% "
              f"{ms:>9.2f} {fps:>9.1f} {model_fps:>9.1f}")
        img_sum += img
        pix_sum += pix
        pro_sum += pro
        count += 1

    if count > 0:
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'MEAN':<14} {img_sum/count:>9.2f}% {pix_sum/count:>9.2f}% {pro_sum/count:>9.2f}%")
    print(f"{'='*100}\n")


def main():
    data_root = ensure_extracted(str(TAR_PATH), str(MVTEC_ROOT))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for backbone in BACKBONES:
        print(f"\n{'#'*100}")
        print(f"  BACKBONE: {backbone}")
        print(f"{'#'*100}\n")

        # Per-backbone results file (resume-safe)
        results_file = REPORTS_DIR / f"cflow_{backbone}_results.json"
        all_results = {}
        if results_file.exists():
            with open(results_file, "r") as f:
                all_results = json.load(f)
            completed = [c for c in ALL_CATEGORIES if c in all_results]
            if len(completed) == len(ALL_CATEGORIES):
                print(f"  All 15 categories already done for {backbone}, skipping.")
                print_summary_table(all_results, backbone)
                continue
            if completed:
                print(f"  Resuming: {len(completed)}/{len(ALL_CATEGORIES)} done.")
                print(f"  Completed: {', '.join(completed)}")

        # Channel cap detection
        channel_cap = get_channel_cap(backbone)

        # Backbone benchmark (once per backbone)
        extractor = build_extractor(backbone, pretrained=True, device=device,
                                    out_indices=CFLOW_OUT_INDICES).eval()
        dummy_input = torch.randn(1, 3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)
        backbone_bench = run_all_benchmarks(extractor, dummy_input, device=device)
        print_benchmark_results(backbone_bench, label=f"Backbone ({backbone})")
        del extractor
        torch.cuda.empty_cache()

        # Run all 15 categories
        for i, cat in enumerate(ALL_CATEGORIES):
            if cat in all_results:
                print(f"\n  [{i+1}/15] {cat}: SKIPPED (already completed)")
                continue

            print(f"\n{'='*80}")
            print(f"  [{i+1}/15] {backbone} / {cat}")
            print(f"{'='*80}\n")

            # Data split
            scan_and_split(
                mvtec_root=Path(data_root),
                out_dir=REPORTS_DIR,
                category=cat,
                val_ratio=VAL_RATIO_CFLOW,
                seed=SEED,
            )
            split_json = REPORTS_DIR / f"mvtec_{cat}_split.json"

            # Data loaders (CFlow: rotate ±5°, drop_last=True)
            train_loader = make_loader_mvtec_ad(
                Path(data_root), cat, "train", split_json,
                input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE,
                rotate_deg=5.0, drop_last_train=True,
            )
            test_loader = make_loader_mvtec_ad(
                Path(data_root), cat, "test", split_json,
                input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE,
            )

            # Run CFlow
            scores, maps, metrics = run_cflow(
                train_loader=train_loader,
                test_loader=test_loader,
                backbone_name=backbone,
                device=device,
                coupling_blocks=8,
                condition_vec=128,
                clamp_alpha=1.9,
                N=256,
                lr=2e-4,
                meta_epochs=25,
                sub_epochs=8,
                input_size=IMAGE_INPUT_SIZE,
                best_metric="pixel",
                backbone_bench=backbone_bench,
                out_indices=CFLOW_OUT_INDICES,
                channel_cap=channel_cap,
            )

            all_results[cat] = metrics

            # Save after each category
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Saved: {results_file}")

            # Free memory between categories
            del scores, maps, metrics
            torch.cuda.empty_cache()

        # Print summary for this backbone
        print_summary_table(all_results, backbone)

    # Final cross-backbone summary
    print(f"\n{'#'*100}")
    print(f"  ALL BACKBONES COMPLETE")
    print(f"{'#'*100}")
    for backbone in BACKBONES:
        results_file = REPORTS_DIR / f"cflow_{backbone}_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                res = json.load(f)
            print_summary_table(res, backbone)


if __name__ == "__main__":
    main()
