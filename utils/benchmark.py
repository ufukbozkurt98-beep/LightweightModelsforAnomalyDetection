"""
Benchmark utilities for measuring model complexity and inference speed.

Provides: parameter count, FLOPs, model size on disk, and inference latency.
Works with any nn.Module — used by FastFlow and CFlow runners.
"""

import time
import tempfile
import os

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters.

    Returns dict with:
        total_params: int
        trainable_params: int
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Count FLOPs for a single forward pass using fvcore.

    Args:
        model: the nn.Module to profile
        input_tensor: example input (single batch)

    Returns:
        Total FLOPs as int.
    """
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total()


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Measure serialized model size in MB by saving to a temp file.
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
    Measure inference latency in milliseconds.

    Uses torch.cuda.Event for GPU timing, time.perf_counter for CPU.

    Args:
        model: nn.Module to benchmark
        input_tensor: example input batch (will be moved to device)
        device: "cpu" or "cuda"
        n_warmup: warmup iterations (not measured)
        n_runs: timed iterations

    Returns dict with:
        mean_ms: float  — average latency per forward pass
        std_ms: float   — standard deviation
        device: str     — device used
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
    Run all benchmarks on a model and return a combined results dict.

    Args:
        model: nn.Module to benchmark
        input_tensor: example input (single batch, e.g. shape [1, 3, 256, 256])
        device: device for latency measurement
        n_warmup: warmup iterations for latency
        n_runs: timed iterations for latency

    Returns dict with:
        total_params, trainable_params, flops, size_mb,
        latency_mean_ms, latency_std_ms, latency_device
    """
    # Parameters
    params = count_parameters(model)

    # FLOPs (always on CPU to avoid device issues with fvcore)
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


def print_benchmark_results(results: dict, label: str = "Model") -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results: {label}")
    print(f"{'=' * 60}")
    print(f"  Total parameters   : {results['total_params']:,}")
    print(f"  Trainable params   : {results['trainable_params']:,}")
    print(f"  FLOPs              : {results['flops']:,}")
    print(f"  Model size         : {results['size_mb']:.2f} MB")
    print(f"  Latency ({results['latency_device']:>4s})    : {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    print(f"{'=' * 60}\n")
