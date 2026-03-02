
from runners.glass_runner import run_glass
from runners.simplenet_runner import run_simplenet

# from glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
# --------------------------------
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


def main():
    tar_path = TAR_PATH  # the place of the mvtec.tar

    # extract the tar_path archive into MVTEC_ROOT, return the extracted path
    data_root = ensure_extracted(tar_path, str(MVTEC_ROOT))

    #  create the folder to store split and report files

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # dataset check and train/val split (STLM handles its own data)
    if METHOD.lower() not in ("stlm",):
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


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STLM has its own data loading (1024x1024, DTD textures, etc.)
    if METHOD.lower() == "stlm":
        # Extract DTD textures if not already extracted
        # DTD tar.gz contains a top-level 'dtd/' folder inside,
        # so we extract into data/ (parent) which creates data/dtd/images/
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
        else:
            print(f"DTD already exists: {dtd_images_dir}")

        from runners.stlm_runner import run_stlm
        metrics = run_stlm(
            category=CATEGORY,
            mvtec_path=str(MVTEC_ROOT),
            dtd_path=str(DTD_ROOT / "images") + "/",
            mobile_sam_path="./weights/mobile_sam.pt",
            sam_vit_h_path="./weights/sam_vit_h_4b8939.pth",
            backbone_key=BACKBONE_KEY,
        )
        return

    # building the data loaders
    train_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "train", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                        batch_size=BATCH_SIZE)
    if METHOD.lower() not in ("cflow", "fastflow"):
        val_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "val", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                          batch_size=BATCH_SIZE)

    test_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "test", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                       batch_size=BATCH_SIZE)


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
            conv3x3_only=True,
            hidden_ratio=1.5,
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
