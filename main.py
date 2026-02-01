from pathlib import Path

import torch

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED,
    IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

from utils.feature_extractor import build_extractor
from methods.cflow_method import CFlowMethod

from utils.eval_metrics_cflow import (
    collect_gt_from_loader, image_level_auroc, pixel_level_auroc, aupro
)
from methods.cflow_train_and_test import train_and_test_cflow


def main():
    # 0) data prepare/split
    data_root = ensure_extracted(TAR_PATH, str(MVTEC_ROOT))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scan_and_split(
        mvtec_root=Path(data_root),
        out_dir=REPORTS_DIR,
        category=CATEGORY,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    # 1) loaders
    train_loader = make_loader_mvtec_ad(
        Path(data_root), CATEGORY, "train", SPLIT_JSON,
        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE
    )
    val_loader = make_loader_mvtec_ad(
        Path(data_root), CATEGORY, "val", SPLIT_JSON,
        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE
    )
    test_loader = make_loader_mvtec_ad(
        Path(data_root), CATEGORY, "test", SPLIT_JSON,
        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE
    )

    # sanity prints
    print("val size:", len(val_loader.dataset))
    b = next(iter(train_loader))
    print("TRAIN shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(val_loader))
    print("VALIDATION shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(test_loader))
    print("TEST  shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())
    print("TEST defect types sample:", b["defect_type"][:4])
    print("Mask sums (per sample):", b["mask"].sum(dim=(1, 2, 3)).tolist())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone_name = "mobilevit_s"
    model_name = "cflow"

    if model_name == "cflow":
        scores, maps, metrics = train_and_test_cflow(
            train_loader=train_loader,
            test_loader=test_loader,
            backbone_name=backbone_name,
            device=device,
            coupling_blocks=8,
            condition_vec=128,
            clamp_alpha=1.9,
            N=256,
            lr=2e-4,
            meta_epochs=25,
            sub_epochs=8,
            input_size=IMAGE_INPUT_SIZE,
        )

        print(f"Image-level AUROC%: {metrics['image_auroc'] * 100:.2f}")
        print(f"Pixel-level AUROC%: {metrics['pixel_auroc'] * 100:.2f}")
        print(f"PRO (AUPRO@0.3)%: {metrics['aupro_0.3'] * 100:.2f}")


if __name__ == "__main__":
    main()
