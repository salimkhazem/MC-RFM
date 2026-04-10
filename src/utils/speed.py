"""Throughput/VRAM measurement helpers """

from __future__ import annotations

import time
from typing import Callable

import torch


@torch.no_grad()
def measure_throughput(fn: Callable[[], torch.Tensor], sample_count: int, device: torch.device) -> tuple[float, float]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    if not torch.isfinite(out).all():
        raise FloatingPointError("Non-finite output during throughput measurement")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    dt = max(time.perf_counter() - t0, 1e-9)
    throughput = float(sample_count / dt)
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return throughput, peak_vram

