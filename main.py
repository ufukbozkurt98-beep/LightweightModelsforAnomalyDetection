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

from utils.eval_metrics import (
    collect_gt_from_loader, image_level_auroc, pixel_level_auroc, aupro
)


def tune_hparams(train_loader, val_loader, device, backbone_name="mobilenetv3_large"):
    """
    Val = good-only olduğu için AUROC ile tune etmiyoruz.
    Objective: normalde false positive düşük olsun.
      obj = q0.999(val_scores) + 0.5*q0.999(val_maps)
    Küçük daha iyi.
    """
    candidates = [
        {"coupling_blocks": 4, "condition_vec": 128, "clamp_alpha": 1.9, "fiber_batch": 2048, "lr": 2e-4},
        {"coupling_blocks": 8, "condition_vec": 128, "clamp_alpha": 1.9, "fiber_batch": 2048, "lr": 2e-4},
        {"coupling_blocks": 8, "condition_vec": 128, "clamp_alpha": 1.9, "fiber_batch": 4096, "lr": 2e-4},
        {"coupling_blocks": 8, "condition_vec": 256, "clamp_alpha": 1.9, "fiber_batch": 4096, "lr": 2e-4},
    ]

    best_cfg = None
    best_obj = float("inf")

    # Tuning hızlı olsun diye epoch düşük tutulur (finalde 100)
    TUNE_EPOCHS = 20

    for cfg in candidates:
        print("\n" + "=" * 80)
        print("[TUNE] trying:", cfg)

        extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()
        model = CFlowMethod(
            extractor,
            device=device,
            coupling_blocks=cfg["coupling_blocks"],
            condition_vec=cfg["condition_vec"],
            clamp_alpha=cfg["clamp_alpha"],
            fiber_batch=cfg["fiber_batch"],
        )

        model.fit(train_loader, epochs=TUNE_EPOCHS, lr=cfg["lr"])

        # val/test aynı scale olsun diye normalizer'ı val üzerinden sabitle
        model.calibrate_normalizer(val_loader)

        val_scores, val_maps = model.predict(val_loader, use_calibrated_max=True)

        q_img = float(torch.quantile(val_scores, 0.999))
        q_pix = float(torch.quantile(val_maps.flatten(), 0.999))
        obj = q_img + 0.5 * q_pix

        print(f"[TUNE] q_img(0.999)={q_img:.4f}  q_pix(0.999)={q_pix:.4f}  obj={obj:.4f}")

        if obj < best_obj:
            best_obj = obj
            best_cfg = cfg

    print("\n" + "=" * 80)
    print("[TUNE] BEST:", best_cfg, "best_obj:", best_obj)
    return best_cfg


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
    backbone_name = "mobilenetv3_large"

    # 2) Hyperparam tuning (val good-only proxy)
    DO_TUNE = False  # True yaparsan hyperparam tuning çalışır
    DO_TRASHHOLD = False
    DEFAULT_CFG = {
        "coupling_blocks": 8,
        "condition_vec": 128,
        "clamp_alpha": 1.9,
        "N": 256,  # ORIJINAL CFLOW-AD
        "lr": 2e-4,
    }

    if DO_TUNE:
        best_cfg = tune_hparams(train_loader, val_loader, device, backbone_name=backbone_name)
    else:
        best_cfg = DEFAULT_CFG
        print("[TUNE] skipped. Using DEFAULT_CFG:", best_cfg)

    # 3) Final train with best cfg (paper-like: 100 epochs)
    print("\n" + "=" * 80)
    print("[FINAL] training with best cfg for 100 epochs:", best_cfg)

    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    cflow = CFlowMethod(
        extractor,
        device=device,
        coupling_blocks=best_cfg["coupling_blocks"],
        condition_vec=best_cfg["condition_vec"],
        clamp_alpha=best_cfg["clamp_alpha"],
        lr=best_cfg["lr"],
        meta_epochs=25,
        sub_epochs=8,
        N=best_cfg["N"],
        input_size=IMAGE_INPUT_SIZE,
    )

    cflow.fit(train_loader)


    img_thr = None
    pix_thr = None

    if DO_TRASHHOLD:
        # 1) scale sabitle
        cflow.calibrate_normalizer(val_loader)

        # 2) threshold seç (good-only)
        val_scores, val_maps = cflow.predict(val_loader, use_calibrated_max=True)
        img_thr = float(torch.quantile(val_scores, 0.995))
        pix_thr = float(torch.quantile(val_maps.flatten(), 0.995))
        print(f"[VAL] img_thr(q0.995)={img_thr:.4f}  pix_thr(q0.995)={pix_thr:.4f}")

        # 3) test aynı scale ile
        scores, maps = cflow.predict(test_loader, use_calibrated_max=True)

    else:
        print("[VAL] skipped calibration + threshold")
        # val kullanılmıyor -> test kendi içinde normalize eder
        scores, maps = cflow.predict(test_loader)

    # 6) Metrics
    y_img, y_pix = collect_gt_from_loader(test_loader)
    img_auc = image_level_auroc(y_img, scores)
    pix_auc = pixel_level_auroc(y_pix, maps)
    pro = aupro(maps, y_pix, expect_fpr=0.3, max_step=1000)

    print(f"Image-level AUROC%: {img_auc * 100:.2f}")
    print(f"Pixel-level AUROC%: {pix_auc * 100:.2f}")
    print(f"PRO (AUPRO@0.3)%: {pro * 100:.2f}")

    # (opsiyonel) threshold ile karar debug
    # labels = torch.from_numpy(y_img)
    # pred = (scores > img_thr).long()
    # acc = (pred == labels).float().mean().item()
    # print(f"[THR DEBUG] acc={acc:.4f}")


if __name__ == "__main__":
    main()
