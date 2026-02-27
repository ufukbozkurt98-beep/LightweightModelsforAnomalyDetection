from configs.config import METHOD

from runners.glass_runner import run_glass
from runners.simplenet_runner import run_simplenet

from glass_src.glass import GLASS
from utils.glass_backbone_adapter import GlassBackboneAdapter
from utils.glass_loader_adapter import GlassLoaderAdapter
from pathlib import Path

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH, BACKBONE_KEY, METHOD, RUN_ALL
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

import torch
import torchvision.models as tvm

from utils.feature_extractor import build_extractor

from configs.config import BACKBONE_KEY

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def run_one_category(category: str, data_root: str):
    """Builds loaders and runs the selected METHOD for a single category."""

    split_json = REPORTS_DIR / f"mvtec_{category}_split.json"

    scan_and_split(
        mvtec_root=Path(data_root),
        out_dir=REPORTS_DIR,
        category=category,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    train_loader = make_loader_mvtec_ad(Path(data_root), category, "train", split_json,
                                        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)
    val_loader   = make_loader_mvtec_ad(Path(data_root), category, "val",   split_json,
                                        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)
    test_loader  = make_loader_mvtec_ad(Path(data_root), category, "test",  split_json,
                                        input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)

    if METHOD.lower() == "glass":
        run_glass(train_loader, val_loader, test_loader)
    elif METHOD.lower() == "simplenet":
        run_simplenet(train_loader, val_loader, test_loader)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")


def main():
    data_root = ensure_extracted(TAR_PATH, str(MVTEC_ROOT))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_ALL:
        # ── loop over every category ──────────────────────────────────────────
        for cat in ALL_CATEGORIES:
            print(f"\n{'='*60}")
            print(f"  Running {METHOD.upper()} on: {cat.upper()}")
            print(f"{'='*60}\n")
            run_one_category(cat, data_root)

    else:
        # ── original single-category behaviour, nothing changed ───────────────
        scan_and_split(
            mvtec_root=Path(data_root),
            out_dir=REPORTS_DIR,
            category=CATEGORY,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        train_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "train", SPLIT_JSON,
                                            input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)
        val_loader   = make_loader_mvtec_ad(Path(data_root), CATEGORY, "val",   SPLIT_JSON,
                                            input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)
        test_loader  = make_loader_mvtec_ad(Path(data_root), CATEGORY, "test",  SPLIT_JSON,
                                            input_size=IMAGE_INPUT_SIZE, batch_size=BATCH_SIZE)

        if METHOD.lower() == "glass":
            run_glass(train_loader, val_loader, test_loader)
        elif METHOD.lower() == "simplenet":
            run_simplenet(train_loader, val_loader, test_loader, category=category)
        else:
            raise ValueError(f"Unknown METHOD: {METHOD}")


if __name__ == "__main__":
    main()
