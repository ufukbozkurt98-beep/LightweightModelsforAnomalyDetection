
from runners.glass_runner import run_glass
from runners.simplenet_runner import run_simplenet

# from glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
# --------------------------------
from pathlib import Path  # to enable to use path objects and /|\ handling

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH,  METHOD, VAL_RATIO_CFLOW
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

import torch

from configs.config import BACKBONE_KEY

from methods.cflow_train_and_test import train_and_test_cflow
from methods.fastflow_train_and_test import train_and_test_fastflow


def main():
    tar_path = TAR_PATH  # the place of the mvtec.tar

    # extract the tar_path archive into MVTEC_ROOT, return the extracted path
    data_root = ensure_extracted(tar_path, str(MVTEC_ROOT))

    #  create the folder to store split and report files

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # dataset check and train/val split
    if METHOD.lower() not in ("cflow", "fastflow"):
        scan_and_split(
            mvtec_root=Path(data_root),
            out_dir=REPORTS_DIR,
            category=CATEGORY,
            val_ratio=VAL_RATIO,
            seed=SEED
        )
    else:
        scan_and_split(
            mvtec_root=Path(data_root),
            out_dir=REPORTS_DIR,
            category=CATEGORY,
            val_ratio= VAL_RATIO_CFLOW,
            seed=SEED
        )

    # building the data loaders
    train_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "train", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                        batch_size=BATCH_SIZE)
    if METHOD.lower() not in ("cflow", "fastflow"):
        val_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "val", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                          batch_size=BATCH_SIZE)

    test_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "test", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                       batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if METHOD.lower() == "glass":
        run_glass(train_loader, val_loader, test_loader)
    elif METHOD.lower() == "simplenet":
        run_simplenet(train_loader, val_loader, test_loader)
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
        )
    elif METHOD.lower() == "fastflow":
        scores, maps, metrics = train_and_test_fastflow(
            train_loader=train_loader,
            test_loader=test_loader,
            backbone_name=BACKBONE_KEY,
            device=device,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            clamp=2.0,
            lr=1e-3,
            meta_epochs=200,
            weight_decay=1e-5,
            input_size=IMAGE_INPUT_SIZE,
        )
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")


if __name__ == "__main__":
    main()
