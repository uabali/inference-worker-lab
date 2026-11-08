from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from inference_lab.api import create_app
from inference_lab.config import Settings
from inference_lab.inference import MockBackend
from inference_lab.models import JobStatus
from inference_lab.queue import JobQueue
from inference_lab.worker import Worker


@pytest.fixture
def client(settings: Settings, queue: JobQueue):
    app = create_app(settings, queue)
    with TestClient(app) as c:
        yield c, queue, settings


def test_healthz(client) -> None:
    c, _, _ = client
    assert c.get("/healthz").json()["status"] == "ok"


def test_enqueue_and_get(client) -> None:
    c, queue, _ = client
    resp = c.post("/v1/jobs", json={"prompt": "hello DS", "max_tokens": 16})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    got = c.get(f"/v1/jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["status"] == "queued"


def test_idempotency_key_is_stable(client) -> None:
    c, queue, _ = client
    body = {"prompt": "once", "max_tokens": 8, "idempotency_key": "abc"}
    first = c.post("/v1/jobs", json=body)
    second = c.post("/v1/jobs", json=body)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    assert queue.stream_length() == 1


def test_queue_full_is_429(settings: Settings, mem) -> None:
    from inference_lab.memory_redis import MemoryRedis

    assert isinstance(mem, MemoryRedis)
    capped = settings.model_copy(update={"max_queue_depth": 1})
    queue = JobQueue(capped, client=mem)
    with TestClient(create_app(capped, queue)) as c:
        assert c.post("/v1/jobs", json={"prompt": "a"}).status_code == 200
        resp = c.post("/v1/jobs", json={"prompt": "b"})
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_worker_processes_job(settings: Settings, queue: JobQueue) -> None:
    settings = settings.model_copy(update={"worker_id": "worker-1", "worker_concurrency": 1, "block_ms": 50})
    job_id = queue.enqueue("process me", max_tokens=4)
    worker = Worker(settings, queue=queue, backend=MockBackend(settings))
    # run one drain cycle manually
    batch = queue.read_batch(settings.worker_id, count=1)
    assert batch
    msg_id, fields = batch[0]
    sem = asyncio.Semaphore(1)
    await worker._handle(msg_id, fields, sem)
    result = queue.get_result(job_id)
    assert result is not None
    assert result.status == JobStatus.succeeded
    assert result.output is not None
    assert "process me" in result.output
