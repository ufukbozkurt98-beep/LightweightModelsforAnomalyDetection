import torch

from utils.feature_extractor import build_extractor
from methods.cflow_method import CFlowMethod
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

    dev = torch.device(device)

    # Build extractor
    extractor = build_extractor(backbone_name, pretrained=True, device=device).eval()

    # Benchmark backbone
    dummy_input = torch.randn(1, 3, input_size, input_size)
    backbone_bench = run_all_benchmarks(extractor, dummy_input, device=device)
    print_benchmark_results(backbone_bench, label=f"Backbone ({backbone_name})")

    # Build cflow
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

    # Collect ground truth early so eval_fn can use it during training
    y_img, y_pix = collect_gt_from_loader(test_loader)

    # Build eval callback: combined (image + pixel) AUROC for best-epoch selection
    def _eval_fn():
        scores, maps = cflow.predict(test_loader)
        pix = pixel_level_auroc(y_pix, maps)
        img = image_level_auroc(y_img, scores)
        combined = (img + pix) / 2
        return combined, img, pix

    # Train with GPU memory tracking
    reset_gpu_peak(dev)
    cflow.fit(train_loader, eval_fn=_eval_fn, eval_every=1)
    gpu_train_mb = measure_gpu_memory_mb(dev)

    # Predict with inference latency measurement
    reset_gpu_peak(dev)
    inference_bench, (scores, maps) = measure_inference_latency(cflow.predict, test_loader, device=device)
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
    print(f"  CFLOW BENCHMARK: {backbone_name}")
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
