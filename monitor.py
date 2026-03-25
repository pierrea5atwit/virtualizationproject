from __future__ import annotations

import threading
import time
from typing import Any, Dict, List


class GPUMonitor:
    """Sample GPU utilization and memory usage in a background thread."""

    def __init__(self, sample_interval_sec: float = 1.0) -> None:
        self.sample_interval_sec = max(float(sample_interval_sec), 0.1)
        self._stop_event = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        initialized = False
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            while not self._stop_event.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                sample = {
                    "timestamp": time.time(),
                    "gpu_util": float(util.gpu),
                    "gpu_mem_mb": float(mem.used / (1024 * 1024)),
                }
                with self._lock:
                    self._samples.append(sample)
                if self._stop_event.wait(self.sample_interval_sec):
                    break
        except Exception:
            # Keep benchmark running even when telemetry setup fails.
            return
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
