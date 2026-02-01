import torch

from utils.feature_extractor import build_extractor
from methods.cflow_method import CFlowMethod
from utils.eval_metrics_cflow import (
    collect_gt_from_loader,
    image_level_auroc,
    pixel_level_auroc,
    aupro,
)


def train_and_test_cflow(
    train_loader,
    test_loader,
    *,
    backbone_name: str = "mobilevit_s",
    device: str | None = None,
    coupling_blocks: int = 8,
    condition_vec: int = 128,
    clamp_alpha: float = 1.9,
    N: int = 256,
    lr: float = 2e-4,
    meta_epochs: int = 25,
    sub_epochs: int = 8,
    input_size: int = 256,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build extractor
    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    # Build CFLOW
    cflow = CFlowMethod(
        extractor,
        device=device,
        coupling_blocks=coupling_blocks,
        condition_vec=condition_vec,
        clamp_alpha=clamp_alpha,
        lr=lr,
        meta_epochs=meta_epochs,
        sub_epochs=sub_epochs,
        N=N,
        input_size=input_size,
    )

    # Train
    cflow.fit(train_loader)

    # Predict
    scores, maps = cflow.predict(test_loader)

    # Results
    y_img, y_pix = collect_gt_from_loader(test_loader)
    img_auc = image_level_auroc(y_img, scores)
    pix_auc = pixel_level_auroc(y_pix, maps)
    pro = aupro(maps, y_pix, expect_fpr=0.3, max_step=1000)

    metrics = {
        "image_auroc": float(img_auc),
        "pixel_auroc": float(pix_auc),
        "aupro_0.3": float(pro),
    }

    return scores, maps, metrics
