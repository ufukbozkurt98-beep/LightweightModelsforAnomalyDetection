from pathlib import Path

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad


def main():
    tar_path = TAR_PATH

    data_root = ensure_extracted(tar_path, str(MVTEC_ROOT))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scan_and_split(
        mvtec_root=Path(data_root),
        out_dir=REPORTS_DIR,
        category=CATEGORY,  # None =  all categories
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    train_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "train", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                        batch_size=BATCH_SIZE)
    val_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "val", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                      batch_size=BATCH_SIZE)
    test_loader = make_loader_mvtec_ad(Path(data_root), CATEGORY, "test", SPLIT_JSON, input_size=IMAGE_INPUT_SIZE,
                                       batch_size=BATCH_SIZE)

    print(len(val_loader.dataset))
    b = next(iter(train_loader))
    print("TRAIN shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(val_loader))
    print("VALIDATION shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(test_loader))
    print("TEST  shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())
    print("TEST defect types sample:", b["defect_type"][:4])

    # confirm at least one mask is non-zero in an anomaly batch
    mask_sums = b["mask"].sum(dim=(1, 2, 3))
    print("Mask sums (per sample):", mask_sums.tolist())


if __name__ == "__main__":
    main()
