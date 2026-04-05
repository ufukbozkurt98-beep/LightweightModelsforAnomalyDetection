"""
model_benchmark.py  –  Model measurement utilities
benchmark.py  –  Model measurement utilities
"""

import os
import time
import tempfile

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
#  GPU memory
# ──────────────────────────────────────────────────────────────────────────────

def reset_gpu_peak(device: torch.device):
    """Reset the peak memory counter so the next read reflects only new usage."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def measure_gpu_memory_mb(device: torch.device) -> float:
    """Return peak GPU memory allocated in MB since the last reset_gpu_peak()."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter count
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


# ──────────────────────────────────────────────────────────────────────────────
#  FLOPs
# ──────────────────────────────────────────────────────────────────────────────

def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Count FLOPs for a single forward pass using fvcore.
    Returns -1 if fvcore is not installed.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total()
    except ImportError:
        print("  [FLOPs skipped: fvcore not installed. Run: pip install fvcore]")
        return -1


# ──────────────────────────────────────────────────────────────────────────────
#  Model size
# ──────────────────────────────────────────────────────────────────────────────

def measure_model_size_mb(model: nn.Module) -> float:
    """Measure serialized model size in MB by saving state_dict to a temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f)
        tmp_path = f.name
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return round(size_mb, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Backbone latency  (single forward pass timing)
# ──────────────────────────────────────────────────────────────────────────────

def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Measure backbone forward-pass latency in milliseconds.
    Uses CUDA events for GPU timing, perf_counter for CPU.
    """
    model.eval()
    model.to(device)
    x = input_tensor.to(device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(x)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(round(times.mean(), 2)),
        "std_ms":  float(round(times.std(),  2)),
        "device":  device,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  End-to-end inference latency  (full test set)
# ──────────────────────────────────────────────────────────────────────────────

def measure_inference_latency(predict_fn, test_loader, device: str = "cpu") -> dict:
    """
    Time how long predict_fn(test_loader) takes over the whole test set.

    predict_fn  — callable that accepts a DataLoader, e.g. sn.predict
    test_loader — the test DataLoader
    device      — "cpu" or "cuda"

    Returns a dict and also the raw outputs from predict_fn.

    Usage:
        latency, scores, maps = measure_inference_latency(sn.predict, test_loader, device="cuda")
    """
    use_cuda   = device.startswith("cuda") and torch.cuda.is_available()
    num_images = len(test_loader.dataset)

    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = predict_fn(test_loader)

    if use_cuda:
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    per_image_ms = (total_s / num_images) * 1000.0

    latency = {
        "total_time_s": round(total_s, 3),
        "num_images":   num_images,
        "per_image_ms": round(per_image_ms, 2),
        "throughput_fps": round(1000.0 / per_image_ms, 1),
    }
    return latency, outputs


# ──────────────────────────────────────────────────────────────────────────────
#  Combined: run all backbone benchmarks at once
# ──────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Run params + FLOPs + model size + latency in one call.
    Returns a combined results dict.

    Usage:
        dummy   = torch.ones(1, 3, 256, 256)
        results = run_all_benchmarks(extractor, dummy, device="cuda")
        print_benchmark_results(results, label="MobileNetV3-Large")
    """
    params  = count_parameters(model)
    flops   = measure_flops(model.cpu().eval(), input_tensor.cpu())
    size_mb = measure_model_size_mb(model)
    latency = measure_latency(model, input_tensor, device=device,
                              n_warmup=n_warmup, n_runs=n_runs)

    return {
        "total_params":     params["total_params"],
        "trainable_params": params["trainable_params"],
        "flops":            flops,
        "size_mb":          size_mb,
        "latency_mean_ms":  latency["mean_ms"],
        "latency_std_ms":   latency["std_ms"],
        "latency_device":   latency["device"],
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Pretty printers
# ──────────────────────────────────────────────────────────────────────────────

def print_benchmark_results(results: dict, label: str = "Model") -> None:
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {label}")
    print(f"{'='*60}")
    print(f"  Total parameters   : {results['total_params']:,}")
    print(f"  Trainable params   : {results['trainable_params']:,}")
    print(f"  FLOPs              : {results['flops']:,}")
    print(f"  Model size         : {results['size_mb']:.2f} MB")
    print(f"  Latency ({results['latency_device']:>4s})    : "
          f"{results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    print(f"{'='*60}\n")


def print_inference_latency(results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Inference Latency ({results.get('device','?')})")
    print(f"{'='*60}")
    print(f"  Total time         : {results['total_time_s']:.3f} s")
    print(f"  Number of images   : {results['num_images']}")
    print(f"  Per-image latency  : {results['per_image_ms']:.2f} ms")
    print(f"  Throughput         : {results['throughput_fps']:.1f} img/s")
    print(f"{'='*60}\n")
