from __future__ import annotations

import time
from statistics import mean
from typing import Any, Dict, List


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def run_worker(worker_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run kNN inference loops and return latency and throughput statistics."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
        from cuml.neighbors import NearestNeighbors  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Missing GPU dependencies. Install RAPIDS/cuML and CuPy in your environment."
        ) from exc

    workload_cfg = config["workload"]
    experiment_cfg = config["experiment"]

    n_samples = int(workload_cfg["n_samples"])
    n_features = int(workload_cfg["n_features"])
    batch_size = int(workload_cfg["batch_size"])
    k_neighbors = int(workload_cfg["k_neighbors"])
    inference_loops = int(workload_cfg["inference_loops"])
    warmup_seconds = float(experiment_cfg.get("warmup_seconds", 0))

    # Build synthetic dataset directly on GPU memory.
    x_train = cp.random.random((n_samples, n_features), dtype=cp.float32)
    model = NearestNeighbors(n_neighbors=k_neighbors)
    model.fit(x_train)

    if warmup_seconds > 0:
        warmup_start = time.perf_counter()
        while (time.perf_counter() - warmup_start) < warmup_seconds:
            q = cp.random.random((batch_size, n_features), dtype=cp.float32)
            model.kneighbors(q, return_distance=False)
        cp.cuda.Stream.null.synchronize()

    latencies_ms: List[float] = []
    total_requests = 0
    loop_start = time.perf_counter()

    for _ in range(inference_loops):
        q = cp.random.random((batch_size, n_features), dtype=cp.float32)
        t0 = time.perf_counter()
        model.kneighbors(q, return_distance=False)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)
        total_requests += batch_size

    total_seconds = max(time.perf_counter() - loop_start, 1e-9)

    return {
        "worker_id": worker_id,
        "latency_ms_mean": float(mean(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_p95": float(_percentile(latencies_ms, 95.0)),
        "throughput_rps": float(total_requests / total_seconds),
        "total_requests": total_requests,
        "duration_seconds": total_seconds,
    }
