from configs.config import METHOD

from runners.glass_runner import run_glass
from runners.simplenet_runner import run_simplenet

from glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
from utils.glass_backbone_adapter import GlassBackboneAdapter
from utils.glass_loader_adapter import GlassLoaderAdapter
#--------------------------------
from pathlib import Path  # to enable to use path objects and /|\ handling

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH, BACKBONE_KEY, METHOD
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

import torch
import torchvision.models as tvm

from utils.feature_extractor import build_extractor

from configs.config import BACKBONE_KEY


def main():
    tar_path = TAR_PATH  # the place of the mvtec.tar

    # extract the tar_path archive into MVTEC_ROOT, return the extracted path
    data_root = ensure_extracted(tar_path, str(MVTEC_ROOT))

    #  create the folder to store split and report files
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # dataset check and train/val split

    scan_and_split(
        mvtec_root=Path(data_root),
        out_dir=REPORTS_DIR,
        category=CATEGORY,  # None =  all categories
        val_ratio=VAL_RATIO,
        seed=SEED
    )


    # building the data loaders
    train_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "train", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                        batch_size=BATCH_SIZE)
    val_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "val", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                      batch_size=BATCH_SIZE)
    test_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "test", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                       batch_size=BATCH_SIZE)

    if METHOD.lower() == "glass":
        run_glass(train_loader, val_loader, test_loader)
    elif METHOD.lower() == "simplenet":
        run_simplenet(train_loader, val_loader, test_loader)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")

if __name__ == "__main__":
    main()
