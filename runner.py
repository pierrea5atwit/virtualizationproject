from __future__ import annotations

import multiprocessing as mp
from queue import Empty
import queue
from typing import Any, Dict, List

from workload import run_worker


def _worker_entry(worker_id: int, config: Dict[str, Any], output_queue: mp.Queue) -> None:
    try:
        output_queue.put({"ok": True, "result": run_worker(worker_id, config)})
    except Exception as exc:
        output_queue.put({"ok": False, "worker_id": worker_id, "error": str(exc)})


def run_workers(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Spawn worker processes and collect per-worker benchmark stats."""
    num_workers = int(config["concurrency"]["num_workers"])
    warmup_seconds = float(config["experiment"].get("warmup_seconds", 0))
    duration_seconds = float(config["experiment"].get("duration_seconds", 0))
    join_timeout_seconds = float(
        config["experiment"].get("join_timeout_seconds", warmup_seconds + duration_seconds + 30.0)
    )

    output_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for worker_id in range(num_workers):
        proc = mp.Process(target=_worker_entry, args=(worker_id, config, output_queue))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join(timeout=join_timeout_seconds)

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, proc in enumerate(processes):
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
            errors.append(f"worker {idx}: timed out after {join_timeout_seconds:.1f}s")

    while len(results) + len(errors) < num_workers:
        try:
            payload = output_queue.get(timeout=1.0)
        except Empty:
            break

        if payload.get("ok"):
            results.append(payload["result"])
        else:
            worker_id = payload.get("worker_id", "unknown")
            errors.append(f"worker {worker_id}: {payload.get('error', 'unknown error')}")

    # Drain late queue messages so they do not leak into the next run.
    while True:
        try:
            output_queue.get_nowait()
        except queue.Empty:
            break

    if errors:
        raise RuntimeError("One or more workers failed: " + "; ".join(errors))

    return sorted(results, key=lambda item: item["worker_id"])
