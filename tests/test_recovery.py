from __future__ import annotations

import asyncio

import pytest

from inference_lab.config import Settings
from inference_lab.inference import InferenceOutput
from inference_lab.models import JobResult, JobStatus
from inference_lab.queue import JobQueue
from inference_lab.worker import Worker


class FlakyBackend:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int) -> InferenceOutput:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return InferenceOutput(text=f"ok:{prompt}", tokens=1, latency_ms=1.0, mode="MOCK")


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int) -> InferenceOutput:
        self.calls += 1
        return InferenceOutput(text=prompt, tokens=1, latency_ms=1.0, mode="MOCK")


async def _drive(worker: Worker, queue: JobQueue, rounds: int) -> None:
    sem = asyncio.Semaphore(1)
    for _ in range(rounds):
        batch = queue.read_batch(worker.settings.worker_id, count=1)
        assert batch, "expected a queued attempt"
        await worker._handle(batch[0][0], batch[0][1], sem)


@pytest.mark.asyncio
async def test_bounded_retry_then_success(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(update={"max_attempts": 3, "worker_id": "worker-1"})
    queue.ensure_group()
    job_id = queue.enqueue("please")
    backend = FlakyBackend(failures=2)
    worker = Worker(settings, queue=queue, backend=backend)  # type: ignore[arg-type]
    await _drive(worker, queue, 3)
    result = queue.get_result(job_id)
    assert result is not None
    assert result.status == JobStatus.succeeded
    assert result.output == "ok:please"
    assert backend.calls == 3
    assert queue.lag() == 0


@pytest.mark.asyncio
async def test_exhausted_attempt_lands_in_dlq(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(update={"max_attempts": 2, "worker_id": "worker-1"})
    queue.ensure_group()
    job_id = queue.enqueue("nope")
    backend = FlakyBackend(failures=10)
    worker = Worker(settings, queue=queue, backend=backend)  # type: ignore[arg-type]
    await _drive(worker, queue, 2)
    result = queue.get_result(job_id)
    assert result is not None and result.status == JobStatus.failed
    assert queue.lag() == 0
    assert queue.r.xlen(settings.dlq_stream_key) == 1
    assert backend.calls == 2


@pytest.mark.asyncio
async def test_reclaim_of_succeeded_result_does_not_rerun(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(update={"worker_id": "worker-2"})
    queue.ensure_group()
    job_id = queue.enqueue("done")
    batch = queue.read_batch("crashed", count=1)
    assert batch
    queue.complete(
        JobResult(job_id=job_id, status=JobStatus.succeeded, prompt="done", output="kept")
    )
    claimed = queue.reclaim("worker-2", count=1, min_idle_ms=0)
    assert claimed
    backend = CountingBackend()
    worker = Worker(settings, queue=queue, backend=backend)  # type: ignore[arg-type]
    sem = asyncio.Semaphore(1)
    await worker._handle(claimed[0][0], claimed[0][1], sem)
    assert backend.calls == 0
    assert queue.get_result(job_id).output == "kept"  # type: ignore[union-attr]
    assert queue.lag() == 0


@pytest.mark.asyncio
async def test_run_forever_reclaims_unacked_message(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(
        update={
            "worker_id": "worker-1",
            "worker_concurrency": 1,
            "claim_idle_ms": 0,
            "block_ms": 20,
            "max_attempts": 3,
        }
    )
    queue.ensure_group()
    job_id = queue.enqueue("orphan")
    crashed = queue.read_batch("crashed-worker", count=1)
    assert crashed and queue.backlog() == (0, 1)
    backend = CountingBackend()
    worker = Worker(settings, queue=queue, backend=backend)  # type: ignore[arg-type]
    task = asyncio.create_task(worker.run_forever())
    status = None
    for _ in range(100):
        result = queue.get_result(job_id)
        if result is not None and result.status == JobStatus.succeeded:
            status = result.status
            break
        await asyncio.sleep(0.02)
    worker.request_stop()
    await asyncio.wait_for(task, timeout=3)
    assert status == JobStatus.succeeded
    assert backend.calls == 1
    assert queue.lag() == 0


@pytest.mark.asyncio
async def test_scaled_down_worker_does_not_pull(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(
        update={"worker_id": "worker-2", "worker_concurrency": 1, "block_ms": 20, "claim_idle_ms": 0}
    )
    queue.ensure_group()
    queue.set_desired_replicas(1)
    job_id = queue.enqueue("leave-me")
    worker = Worker(settings, queue=queue, backend=CountingBackend())  # type: ignore[arg-type]
    task = asyncio.create_task(worker.run_forever())
    await asyncio.sleep(0.15)
    worker.request_stop()
    await asyncio.wait_for(task, timeout=3)
    result = queue.get_result(job_id)
    assert result is not None and result.status == JobStatus.queued
    assert queue.backlog() == (1, 0)
