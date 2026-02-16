import torch

from utils.feature_extractor import build_extractor
from methods.fastflow_method import FastFlowMethod
from utils.eval_metrics_cflow import (
    collect_gt_from_loader,
    image_level_auroc,
    pixel_level_auroc,
    aupro,
)
from utils.benchmark import run_all_benchmarks, print_benchmark_results, measure_inference_latency, print_inference_latency


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
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build extractor
    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    # Benchmark backbone
    dummy_input = torch.randn(1, 3, input_size, input_size)
    backbone_bench = run_all_benchmarks(extractor, dummy_input, device=device)
    print_benchmark_results(backbone_bench, label=f"Backbone ({backbone_name})")

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

    # Predict with inference latency measurement
    inference_bench, scores, maps = measure_inference_latency(fastflow.predict, test_loader, device=device)
    print_inference_latency(inference_bench, device=device)

    # Results
    y_img, y_pix = collect_gt_from_loader(test_loader)
    img_auc = image_level_auroc(y_img, scores)
    pix_auc = pixel_level_auroc(y_pix, maps)
    pro = aupro(maps, y_pix, expect_fpr=0.3, max_step=1000)

    metrics = {
        "image_auroc": float(img_auc),
        "pixel_auroc": float(pix_auc),
        "aupro_0.3": float(pro),
        "backbone_benchmark": backbone_bench,
        "inference_benchmark": inference_bench,
    }
    print(f"Image-level AUROC%: {metrics['image_auroc'] * 100:.2f}")
    print(f"Pixel-level AUROC%: {metrics['pixel_auroc'] * 100:.2f}")
    print(f"PRO (AUPRO@0.3)%: {metrics['aupro_0.3'] * 100:.2f}")

    return scores, maps, metrics
