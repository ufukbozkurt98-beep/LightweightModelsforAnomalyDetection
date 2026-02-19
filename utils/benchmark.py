import time
import tempfile
import os

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Count FLOPs for a single forward pass using fvcore
    """
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total()


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Measure serialized model size in MB by saving to a temp file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f)
        tmp_path = f.name
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Measure inference latency in milliseconds
    """
    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            _ = model(x)

        if use_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(x)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # ms
            else:
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # ms

    import numpy as np
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "device": device,
    }


def run_all_benchmarks(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Run all benchmarks on a model and return a combined results dict
    """
    # Parameters
    params = count_parameters(model)

    # FLOPs
    model_cpu = model.cpu().eval()
    input_cpu = input_tensor.cpu()
    flops = measure_flops(model_cpu, input_cpu)

    # Model size
    size_mb = measure_model_size_mb(model)

    # Latency
    latency = measure_latency(model, input_tensor, device=device, n_warmup=n_warmup, n_runs=n_runs)

    return {
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "flops": flops,
        "size_mb": round(size_mb, 2),
        "latency_mean_ms": round(latency["mean_ms"], 2),
        "latency_std_ms": round(latency["std_ms"], 2),
        "latency_device": latency["device"],
    }


def measure_inference_latency(predict_fn, test_loader, device: str = "cpu") -> dict:
    """
    Measure real end-to-end inference latency by timing the predict function.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Count total images in test set
    num_images = len(test_loader.dataset)

    if use_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    scores, maps = predict_fn(test_loader)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_s = t1 - t0
    per_image_ms = (total_s / num_images) * 1000.0

    return {
        "total_time_s": round(total_s, 3),
        "num_images": num_images,
        "per_image_ms": round(per_image_ms, 2),
    }, scores, maps


def print_inference_latency(results: dict, device: str = "cpu") -> None:
    """print inference latency results"""
    print(f"\n{'=' * 60}")
    print(f"Inference Latency ({device})")
    print(f"{'=' * 60}")
    print(f"  Total time         : {results['total_time_s']:.3f} s")
    print(f"  Number of images   : {results['num_images']}")
    print(f"  Per-image latency  : {results['per_image_ms']:.2f} ms")
    print(f"  Throughput         : {1000.0 / results['per_image_ms']:.1f} img/s")
    print(f"{'=' * 60}\n")


def print_benchmark_results(results: dict, label: str = "Model") -> None:
    """print benchmark results"""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results: {label}")
    print(f"{'=' * 60}")
    print(f"  Total parameters   : {results['total_params']:,}")
    print(f"  Trainable params   : {results['trainable_params']:,}")
    print(f"  FLOPs              : {results['flops']:,}")
    print(f"  Model size         : {results['size_mb']:.2f} MB")
    print(f"  Latency ({results['latency_device']:>4s})    : {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    print(f"{'=' * 60}\n")
