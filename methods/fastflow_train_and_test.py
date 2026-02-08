import torch

from utils.feature_extractor import build_extractor
from methods.fastflow_method import FastFlowMethod
from utils.eval_metrics_cflow import (
    collect_gt_from_loader,
    image_level_auroc,
    pixel_level_auroc,
    aupro,
)


def train_and_test_fastflow(
    train_loader,
    test_loader,
    *,
    backbone_name: str = "mobilenetv3_large",
    device: str | None = None,
    flow_steps: int = 8,
    conv3x3_only: bool = False,
    hidden_ratio: float = 1.0,
    clamp: float = 2.0,
    lr: float = 1e-3,
    meta_epochs: int = 50,
    weight_decay: float = 1e-5,
    input_size: int = 256,
):
    """
    Train and test FastFlow, return scores, anomaly maps and metrics.

    Parameters follow the same naming convention as train_and_test_cflow().
    FastFlow-specific parameters (flow_steps, conv3x3_only, hidden_ratio, clamp)
    follow the anomalib FastflowModel defaults.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build extractor (same as cflow_train_and_test.py)
    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    # Build FastFlow
    fastflow = FastFlowMethod(
        extractor,
        device=device,
        input_size=input_size,
        flow_steps=flow_steps,
        conv3x3_only=conv3x3_only,
        hidden_ratio=hidden_ratio,
        clamp=clamp,
        lr=lr,
        meta_epochs=meta_epochs,
        weight_decay=weight_decay,
    )

    # Train
    fastflow.fit(train_loader)

    # Predict
    scores, maps = fastflow.predict(test_loader)

    # Results (same metric computation as cflow_train_and_test.py)
    y_img, y_pix = collect_gt_from_loader(test_loader)
    img_auc = image_level_auroc(y_img, scores)
    pix_auc = pixel_level_auroc(y_pix, maps)
    pro = aupro(maps, y_pix, expect_fpr=0.3, max_step=1000)

    metrics = {
        "image_auroc": float(img_auc),
        "pixel_auroc": float(pix_auc),
        "aupro_0.3": float(pro),
    }
    print(f"Image-level AUROC%: {metrics['image_auroc'] * 100:.2f}")
    print(f"Pixel-level AUROC%: {metrics['pixel_auroc'] * 100:.2f}")
    print(f"PRO (AUPRO@0.3)%: {metrics['aupro_0.3'] * 100:.2f}")

    return scores, maps, metrics
