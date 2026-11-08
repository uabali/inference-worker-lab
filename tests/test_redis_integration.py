"""Tests against a real Redis. Skips only when nothing is listening.

Start one with:
  redis-server --bind 127.0.0.1 --port 6399 --save "" --appendonly no
"""

from __future__ import annotations

import asyncio
import os

import pytest
import redis

from inference_lab.config import Settings
from inference_lab.inference import MockBackend
from inference_lab.models import JobStatus
from inference_lab.queue import JobQueue
from inference_lab.worker import Worker

REDIS_URL = os.environ.get("IWL_TEST_REDIS_URL", "redis://127.0.0.1:6399/0")


@pytest.fixture
def redis_queue() -> JobQueue:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip(f"redis not reachable at {REDIS_URL}")
    client.flushdb()
    settings = Settings(
        redis_url=REDIS_URL,
        inference_mode="MOCK",
        mock_latency_ms=1,
        mock_latency_jitter_ms=0,
        worker_id="worker-1",
        worker_concurrency=1,
        claim_idle_ms=50,
        block_ms=50,
        max_attempts=3,
        max_queue_depth=100,
        scale_min=1,
        scale_max=4,
        scale_up_lag=5,
        scale_down_lag=1,
        hysteresis_hold_ticks=2,
    )
    queue = JobQueue(settings, client=client)
    yield queue
    client.flushdb()
    client.close()


def test_real_redis_reclaim_idle_pending(redis_queue: JobQueue) -> None:
    job_id = redis_queue.enqueue("stay")
    batch = redis_queue.read_batch("crashed-worker", count=1)
    assert batch and batch[0][1]["job_id"] == job_id
    assert redis_queue.reclaim("worker-1", count=5, min_idle_ms=60_000) == []
    claimed = redis_queue.reclaim("worker-1", count=5, min_idle_ms=0)
    assert len(claimed) == 1
    assert claimed[0][1]["job_id"] == job_id
    never, pending = redis_queue.backlog()
    assert never == 0
    assert pending == 1


@pytest.mark.asyncio
async def test_real_redis_worker_finishes_crashed_claim(redis_queue: JobQueue) -> None:
    job_id = redis_queue.enqueue("recover me")
    crashed = redis_queue.read_batch("crashed-worker", count=1)
    assert crashed
    # Idle long enough for claim_idle_ms=50 before the live worker starts.
    await asyncio.sleep(0.12)
    worker = Worker(redis_queue.settings, queue=redis_queue, backend=MockBackend(redis_queue.settings))
    task = asyncio.create_task(worker.run_forever())
    status = None
    for _ in range(100):
        result = redis_queue.get_result(job_id)
        if result is not None and result.status == JobStatus.succeeded:
            status = result.status
            assert result.output is not None and "recover me" in result.output
            break
        await asyncio.sleep(0.03)
    worker.request_stop()
    await asyncio.wait_for(task, timeout=3)
    assert status == JobStatus.succeeded
    assert redis_queue.lag() == 0
