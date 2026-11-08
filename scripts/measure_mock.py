#!/usr/bin/env python3
"""In-process MOCK measurement.

Not a GPU run, not a network hop, not an SLO. Prints one sample on this machine.
Queue wait is finished_at - created_at. backend_latency_ms is the mock sleep.
"""

from __future__ import annotations

import asyncio
import os
import platform
import time

from inference_lab.config import Settings
from inference_lab.inference import MockBackend
from inference_lab.memory_redis import MemoryRedis
from inference_lab.models import JobStatus
from inference_lab.queue import JobQueue
from inference_lab.worker import Worker


def _pct(values: list[float], p: float) -> float:
    if not values:
        raise RuntimeError("no samples")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def main() -> None:
    n = 80
    settings = Settings(
        inference_mode="MOCK",
        mock_latency_ms=5,
        mock_latency_jitter_ms=0,
        worker_id="worker-1",
        worker_concurrency=4,
        block_ms=20,
        claim_idle_ms=30_000,
        max_queue_depth=0,
        max_attempts=3,
    )
    queue = JobQueue(settings, client=MemoryRedis())  # type: ignore[arg-type]
    job_ids = [queue.enqueue(f"job {i}", max_tokens=8) for i in range(n)]
    worker = Worker(settings, queue=queue, backend=MockBackend(settings))
    started = time.perf_counter()
    task = asyncio.create_task(worker.run_forever())
    deadline = time.time() + 30
    while time.time() < deadline:
        done = sum(
            1
            for job_id in job_ids
            if (result := queue.get_result(job_id)) is not None and result.status != JobStatus.queued
            and result.status != JobStatus.running
        )
        if done == n:
            break
        await asyncio.sleep(0.01)
    elapsed = time.perf_counter() - started
    worker.request_stop()
    await asyncio.wait_for(task, timeout=5)

    e2e: list[float] = []
    backend: list[float] = []
    succeeded = 0
    for job_id in job_ids:
        result = queue.get_result(job_id)
        if result is None or result.status != JobStatus.succeeded:
            continue
        succeeded += 1
        if result.latency_ms is not None:
            backend.append(result.latency_ms)
        if result.created_at is not None and result.finished_at is not None:
            e2e.append((result.finished_at - result.created_at) * 1000.0)

    print(f"platform={platform.platform()}")
    print(f"processor={platform.processor() or 'unknown'} machine={platform.machine()} cpu_count={os.cpu_count()}")
    print(
        f"n={n} concurrency={settings.worker_concurrency} mock_latency_ms=5 jitter_ms=0 "
        "redis=MemoryRedis gpu=no"
    )
    print(f"wall_s={elapsed:.4f} succeeded={succeeded} failed={n - succeeded}")
    print(
        "backend_latency_ms "
        f"p50={_pct(backend, 50):.2f} p95={_pct(backend, 95):.2f} p99={_pct(backend, 99):.2f} "
        "method=nearest-rank"
    )
    print(
        "queue_to_done_ms "
        f"p50={_pct(e2e, 50):.2f} p95={_pct(e2e, 95):.2f} p99={_pct(e2e, 99):.2f} "
        "method=nearest-rank"
    )


if __name__ == "__main__":
    asyncio.run(main())
