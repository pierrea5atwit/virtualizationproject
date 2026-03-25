from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
from statistics import mean
from typing import Any, Dict, List

from config import load_config
from hardware import detect_hardware_profile, validate_environment_request
from logger import CSVExperimentLogger
from monitor import GPUMonitor
from runner import run_workers


def _aggregate_metrics(worker_rows: List[Dict[str, Any]], monitor_samples: List[Dict[str, Any]]) -> Dict[str, float]:
    latency_ms = mean([row["latency_ms_mean"] for row in worker_rows]) if worker_rows else 0.0
    throughput_rps = sum([row["throughput_rps"] for row in worker_rows])

    if monitor_samples:
        gpu_util = mean([s["gpu_util"] for s in monitor_samples])
        gpu_mem = mean([s["gpu_mem_mb"] for s in monitor_samples])
    else:
        gpu_util = 0.0
        gpu_mem = 0.0

    return {
        "latency_ms": float(latency_ms),
        "throughput_rps": float(throughput_rps),
        "gpu_util": float(gpu_util),
        "gpu_mem": float(gpu_mem),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU contention benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def _resolve_worker_counts(config: Dict[str, Any]) -> List[int]:
    concurrency_cfg = config["concurrency"]
    if "worker_levels" in concurrency_cfg:
        raw = concurrency_cfg["worker_levels"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("concurrency.worker_levels must be a non-empty list of integers")
        return [int(v) for v in raw]
    return [int(concurrency_cfg["num_workers"])]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = config["experiment"]
    monitoring_cfg = config["monitoring"]
    logging_cfg = config["logging"]

    environment = config["environment"]
    repeat_runs = int(experiment_cfg["repeat_runs"])
    worker_counts = _resolve_worker_counts(config)

    logger = CSVExperimentLogger(logging_cfg["output_dir"])

    hardware_profile = detect_hardware_profile()
    validation = validate_environment_request(hardware_profile, environment)
    logger.log_hardware(
        {
            "environment": environment,
            "has_nvidia_gpu": hardware_profile.has_nvidia_gpu,
            "gpu_name": hardware_profile.gpu_name,
            "virtualization_mode": hardware_profile.virtualization_mode,
            "source": hardware_profile.source,
            "validation_ok": validation["ok"],
            "validation_reason": validation["reason"],
            "error": hardware_profile.error,
        }
    )

    print(
        "hardware "
        f"has_nvidia_gpu={hardware_profile.has_nvidia_gpu} "
        f"gpu_name='{hardware_profile.gpu_name or 'unknown'}' "
        f"virtualization_mode={hardware_profile.virtualization_mode}"
    )

    if not bool(validation["ok"]):
        raise RuntimeError(str(validation["reason"]))

    for num_workers in worker_counts:
        run_config = copy.deepcopy(config)
        run_config["concurrency"]["num_workers"] = num_workers

        for run_index in range(repeat_runs):
            monitor = GPUMonitor(sample_interval_sec=monitoring_cfg.get("sample_interval_sec", 1))
            monitor.start()

            worker_rows = run_workers(run_config)

            monitor.stop()
            metrics = _aggregate_metrics(worker_rows, monitor.samples())

            logger.log_workers(environment=environment, run_index=run_index, worker_rows=worker_rows)
            logger.log_summary(
                {
                    "environment": environment,
                    "run_index": run_index,
                    "num_workers": num_workers,
                    "latency_ms": metrics["latency_ms"],
                    "throughput_rps": metrics["throughput_rps"],
                    "gpu_util": metrics["gpu_util"],
                    "gpu_mem": metrics["gpu_mem"],
                }
            )

            print(
                f"run={run_index} env={environment} workers={num_workers} "
                f"latency_ms={metrics['latency_ms']:.2f} throughput_rps={metrics['throughput_rps']:.2f} "
                f"gpu_util={metrics['gpu_util']:.2f}% gpu_mem={metrics['gpu_mem']:.2f}MB"
            )

            if metrics["gpu_util"] < 30.0:
                print("warning: low GPU utilization detected (<30%); workload may be CPU-bound")
            if metrics["throughput_rps"] <= 0:
                print("warning: non-positive throughput detected; check worker errors and contention level")


if __name__ == "__main__":
    mp.freeze_support()
    main()
