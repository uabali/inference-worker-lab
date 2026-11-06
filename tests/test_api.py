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
