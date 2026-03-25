from __future__ import annotations

import multiprocessing as mp
from queue import Empty
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

    output_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for worker_id in range(num_workers):
        proc = mp.Process(target=_worker_entry, args=(worker_id, config, output_queue))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

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

    if errors:
        raise RuntimeError("One or more workers failed: " + "; ".join(errors))

    return sorted(results, key=lambda item: item["worker_id"])
