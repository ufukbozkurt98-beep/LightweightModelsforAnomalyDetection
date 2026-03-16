
from runners.glass_runner import run_glass
from runners.simplenet_runner import run_simplenet

# from glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
# --------------------------------
import json
from pathlib import Path  # to enable to use path objects and /|\ handling

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH,  METHOD, VAL_RATIO_CFLOW,
    DTD_ZIP_PATH, DTD_ROOT
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

import torch

from configs.config import BACKBONE_KEY

from methods.cflow_train_and_test import train_and_test_cflow
from methods.fastflow_train_and_test import train_and_test_fastflow


ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def run_single_category(category, data_root, device, backbone_bench=None, cflow_out_indices=None, channel_cap=None):
    """Run training and evaluation for a single MVTec AD category."""
    split_json = REPORTS_DIR / f"mvtec_{category}_split.json"

    # dataset check and train/val split (STLM handles its own data)
    if METHOD.lower() not in ("stlm",):
        val_ratio = VAL_RATIO_CFLOW if METHOD.lower() in ("cflow", "fastflow") else VAL_RATIO
        scan_and_split(
            mvtec_root=Path(data_root),
            out_dir=REPORTS_DIR,
            category=category,
            val_ratio=val_ratio,
            seed=SEED
        )

    # STLM has its own data loading
    if METHOD.lower() == "stlm":
        dtd_images_dir = DTD_ROOT / "images"
        if not dtd_images_dir.exists():
            import tarfile, zipfile
            archive_path = Path(DTD_ZIP_PATH)
            if not archive_path.exists():
                raise FileNotFoundError(f"DTD archive not found: {archive_path}")
            print(f"Extracting DTD: {archive_path} -> {DTD_ROOT.parent}")
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tar:
                    tar.extractall(path=str(DTD_ROOT.parent))
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(path=str(DTD_ROOT.parent))
            print(f"DTD extracted to: {DTD_ROOT}")

        from runners.stlm_runner import run_stlm
        metrics = run_stlm(
            category=category,
            mvtec_path=str(MVTEC_ROOT),
            dtd_path=str(DTD_ROOT / "images") + "/",
            mobile_sam_path="./weights/mobile_sam.pt",
            sam_vit_h_path="./weights/sam_vit_h_4b8939.pth",
            backbone_key=BACKBONE_KEY,
            backbone_bench=backbone_bench,
        )
        return metrics

    # Build data loaders
    # CFlow-AD original uses RandomRotation(5) and drop_last=True for training
    is_cflow = METHOD.lower() == "cflow"
    train_loader = make_loader_mvtec_ad(Path(data_root), category, "train", split_json,
                                        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE,
                                        rotate_deg=5.0 if is_cflow else 0.0,
                                        drop_last_train=True if is_cflow else False)
    if METHOD.lower() not in ("cflow", "fastflow"):
        val_loader = make_loader_mvtec_ad(Path(data_root), category, "val", split_json,
                                          input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)
    test_loader = make_loader_mvtec_ad(Path(data_root), category, "test", split_json,
                                       input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)

    if METHOD.lower() == "glass":
        run_glass(train_loader, val_loader, test_loader)
        return None
    elif METHOD.lower() == "simplenet":
        run_simplenet(train_loader, val_loader, test_loader)
        return None
    elif METHOD.lower() == "cflow":
        scores, maps, metrics = train_and_test_cflow(
            train_loader=train_loader,
            test_loader=test_loader,
            backbone_name=BACKBONE_KEY,
            device=device,
            coupling_blocks=8,
            condition_vec=128,
            clamp_alpha=1.9,
            N=256,
            lr=2e-4,
            meta_epochs=25,
            sub_epochs=8,
            input_size=IMAGE_INPUT_SIZE,
            best_metric="combined",  # "pixel" = original CFlow paper, "combined" = (img+pix)/2
            backbone_bench=backbone_bench,
            out_indices=cflow_out_indices,
            channel_cap=channel_cap,
        )
        return metrics
    elif METHOD.lower() == "fastflow":
        scores, maps, metrics = train_and_test_fastflow(
            train_loader=train_loader,
            test_loader=test_loader,
            backbone_name=BACKBONE_KEY,
            device=device,
            flow_steps=8,
            conv3x3_only=True,
            hidden_ratio=1.5,
            clamp=2.0,
            lr=1e-3,
            meta_epochs=200,
            weight_decay=1e-5,
            input_size=IMAGE_INPUT_SIZE,
            backbone_bench=backbone_bench,
            # Enhancement toggles (set to False/0.0 to match vanilla anomalib)
            zero_init=False,
            gauss_sigma=4.0,
            use_scheduler=True,
            channel_cap=channel_cap,
        )
        return metrics
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")


def print_summary_table(all_results, method, backbone):
    """Print a summary table of results across all categories."""
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {method.upper()} + {backbone}")
    print(f"{'='*80}")
    print(f"  {'Category':<14} {'Img AUROC':>10} {'Pix AUROC':>10} {'AUPRO':>10} {'ms/img':>10} {'FPS':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    img_sum, pix_sum, pro_sum = 0.0, 0.0, 0.0
    count = 0

    for cat, m in all_results.items():
        if m is None:
            print(f"  {cat:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        img = m.get("image_auroc", 0) * 100
        pix = m.get("pixel_auroc", 0) * 100
        pro = m.get("aupro_0.3", 0) * 100
        ms = m.get("inference_benchmark", {}).get("per_image_ms", 0)
        fps = m.get("inference_benchmark", {}).get("throughput_fps", 0)
        print(f"  {cat:<14} {img:>9.2f}% {pix:>9.2f}% {pro:>9.2f}% {ms:>9.2f} {fps:>9.1f}")
        img_sum += img
        pix_sum += pix
        pro_sum += pro
        count += 1

    if count > 0:
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'MEAN':<14} {img_sum/count:>9.2f}% {pix_sum/count:>9.2f}% {pro_sum/count:>9.2f}%")
    print(f"{'='*80}\n")


def main():
    from utils.feature_extractor import build_extractor
    from utils.model_benchmark import run_all_benchmarks, print_benchmark_results

    tar_path = TAR_PATH
    data_root = ensure_extracted(tar_path, str(MVTEC_ROOT))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine categories to run
    if CATEGORY.lower() == "all":
        categories = ALL_MVTEC_CATEGORIES
    else:
        categories = [CATEGORY]

    # CFlow-AD original paper uses deeper feature layers: features[-11,-5,-2]
    # which corresponds to out_indices=(2,3,4) in timm → 32×32, 16×16, 8×8
    cflow_out_indices = (2, 3, 4) if METHOD.lower() == "cflow" else None

    # Auto-detect channel_cap for wide-channel backbones (e.g. ShuffleNet 240/480/960).
    # NF-based methods (CFlow, FastFlow) struggle with >256 channels per scale.
    # Adds a 1x1 conv to reduce channels before the normalizing flow.
    channel_cap = None
    if METHOD.lower() in ("cflow", "fastflow"):
        from utils.feature_extractor import build_extractor as _be
        _tmp = _be(BACKBONE_KEY, pretrained=False)
        max_ch = max(_tmp.feature_channels.values())
        if max_ch > 512:
            channel_cap = 256
            print(f"  Auto channel_cap={channel_cap} (backbone max channel={max_ch} > 512)")
        del _tmp

    # Run backbone benchmark once before the category loop
    backbone_bench = None
    if METHOD.lower() in ("cflow", "fastflow"):
        extractor = build_extractor(BACKBONE_KEY, pretrained=True, device=device, out_indices=cflow_out_indices).eval()
        dummy_input = torch.randn(1, 3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)
        backbone_bench = run_all_benchmarks(extractor, dummy_input, device=device)
        print_benchmark_results(backbone_bench, label=f"Backbone ({BACKBONE_KEY})")
        del extractor  # free memory, each category builds its own
    elif METHOD.lower() == "stlm":
        from stlm_code.mob_sam import BackboneEncoderAdapter
        adapter = BackboneEncoderAdapter(BACKBONE_KEY, pretrained=True).eval()
        dummy_input = torch.randn(1, 3, 1024, 1024)
        backbone_bench = run_all_benchmarks(adapter, dummy_input, device=device)
        print_benchmark_results(backbone_bench, label=f"STLM Encoder ({BACKBONE_KEY})")
        del adapter

    # Resume support: save results after each category so interrupted runs can continue.
    # Results file: data/reports/{method}_{backbone}_results.json
    results_file = REPORTS_DIR / f"{METHOD.lower()}_{BACKBONE_KEY}_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
        completed = [c for c in categories if c in all_results]
        if completed:
            print(f"\n  Resuming: {len(completed)}/{len(categories)} categories already done.")
            print(f"  Completed: {', '.join(completed)}")
            print(f"  Results file: {results_file}")

    for i, cat in enumerate(categories):
        if cat in all_results:
            print(f"\n  [{i+1}/{len(categories)}] {cat}: SKIPPED (already completed)")
            continue

        print(f"\n{'#'*80}")
        print(f"  [{i+1}/{len(categories)}] Category: {cat}")
        print(f"{'#'*80}\n")

        metrics = run_single_category(cat, data_root, device, backbone_bench=backbone_bench, cflow_out_indices=cflow_out_indices, channel_cap=channel_cap)
        all_results[cat] = metrics

        # Save after each category so progress is not lost
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {results_file}")

    # Print summary table if multiple categories were run
    if len(categories) > 1:
        print_summary_table(all_results, METHOD, BACKBONE_KEY)


if __name__ == "__main__":
    main()
