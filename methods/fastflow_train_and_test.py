import torch

from utils.feature_extractor import build_extractor
from methods.fastflow_method import FastFlowMethod
from utils.eval_metrics_cflow import (
    collect_gt_from_loader,
    image_level_auroc,
    pixel_level_auroc,
    aupro,
)
from utils.model_benchmark import (
    run_all_benchmarks, print_benchmark_results,
    reset_gpu_peak, measure_gpu_memory_mb,
    measure_inference_latency,
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
    backbone_bench: dict | None = None,
):
    """
    Train and test FastFlow, return scores, anomaly maps and metrics.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)

    # Build extractor
    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    # Benchmark backbone (skip if pre-computed)
    if backbone_bench is None:
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

    # Collect ground truth early so eval_fn can use it during training
    y_img, y_pix = collect_gt_from_loader(test_loader)

    # Build eval callback: combined (image + pixel) AUROC for best-epoch selection
    def _eval_fn():
        scores, maps = fastflow.predict(test_loader)
        pix = pixel_level_auroc(y_pix, maps)
        img = image_level_auroc(y_img, scores)
        combined = (img + pix) / 2
        return combined, img, pix

    # Train with GPU memory tracking
    reset_gpu_peak(dev)
    fastflow.fit(train_loader, eval_fn=_eval_fn, eval_every=10)
    gpu_train_mb = measure_gpu_memory_mb(dev)

    # Predict with inference latency measurement
    reset_gpu_peak(dev)
    inference_bench, (scores, maps) = measure_inference_latency(fastflow.predict, test_loader, device=device)
    gpu_infer_mb = measure_gpu_memory_mb(dev)

    # Results
    img_auc = image_level_auroc(y_img, scores)
    pix_auc = pixel_level_auroc(y_pix, maps)
    pro = aupro(maps, y_pix, expect_fpr=0.3, max_step=1000)

    metrics = {
        "image_auroc": float(img_auc),
        "pixel_auroc": float(pix_auc),
        "aupro_0.3": float(pro),
        "backbone_benchmark": backbone_bench,
        "inference_benchmark": inference_bench,
        "gpu_train_mb": round(gpu_train_mb, 1),
        "gpu_infer_mb": round(gpu_infer_mb, 1),
    }

    print(f"\n{'='*55}")
    print(f"  FASTFLOW BENCHMARK: {backbone_name}")
    print(f"{'='*55}")
    print(f"  Image AUROC  : {metrics['image_auroc'] * 100:.2f}%")
    print(f"  Pixel AUROC  : {metrics['pixel_auroc'] * 100:.2f}%")
    print(f"  PRO          : {metrics['aupro_0.3'] * 100:.2f}%")
    print(f"  GPU (train)  : {gpu_train_mb:.0f} MB")
    print(f"  GPU (infer)  : {gpu_infer_mb:.0f} MB")
    print(f"  Infer total  : {inference_bench['total_time_s']:.3f} s")
    print(f"  Infer/image  : {inference_bench['per_image_ms']:.2f} ms")
    print(f"  Throughput   : {inference_bench['throughput_fps']:.1f} FPS")
    print(f"{'='*55}\n")

    return scores, maps, metrics
