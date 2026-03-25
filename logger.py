from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable


class CSVExperimentLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.csv"
        self.workers_path = self.output_dir / "workers.csv"
        self.hardware_path = self.output_dir / "hardware.csv"

        if not self.summary_path.exists():
            self._write_header(
                self.summary_path,
                [
                    "timestamp",
                    "environment",
                    "run_index",
                    "num_workers",
                    "latency_ms",
                    "throughput_rps",
                    "gpu_util",
                    "gpu_mem",
                ],
            )

        if not self.workers_path.exists():
            self._write_header(
                self.workers_path,
                [
                    "timestamp",
                    "environment",
                    "run_index",
                    "worker_id",
                    "latency_ms_mean",
                    "latency_ms_p95",
                    "throughput_rps",
                    "duration_seconds",
                    "total_requests",
                ],
            )

        if not self.hardware_path.exists():
            self._write_header(
                self.hardware_path,
                [
                    "timestamp",
                    "environment",
                    "has_nvidia_gpu",
                    "gpu_name",
                    "virtualization_mode",
                    "source",
                    "validation_ok",
                    "validation_reason",
                    "error",
                ],
            )

    @staticmethod
    def _write_header(path: Path, columns: Iterable[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(columns))

    def log_summary(self, row: Dict[str, Any]) -> None:
        with self.summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(time.time()),
                    row["environment"],
                    row["run_index"],
                    row["num_workers"],
                    f"{row['latency_ms']:.4f}",
                    f"{row['throughput_rps']:.4f}",
                    f"{row['gpu_util']:.4f}",
                    f"{row['gpu_mem']:.4f}",
                ]
            )

    def log_workers(self, environment: str, run_index: int, worker_rows: Iterable[Dict[str, Any]]) -> None:
        with self.workers_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in worker_rows:
                writer.writerow(
                    [
                        int(time.time()),
                        environment,
                        run_index,
                        row["worker_id"],
                        f"{row['latency_ms_mean']:.4f}",
                        f"{row['latency_ms_p95']:.4f}",
                        f"{row['throughput_rps']:.4f}",
                        f"{row['duration_seconds']:.4f}",
                        row["total_requests"],
                    ]
                )

    def log_hardware(self, row: Dict[str, Any]) -> None:
        with self.hardware_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(time.time()),
                    row["environment"],
                    row["has_nvidia_gpu"],
                    row["gpu_name"],
                    row["virtualization_mode"],
                    row["source"],
                    row["validation_ok"],
                    row["validation_reason"],
                    row["error"],
                ]
            )
