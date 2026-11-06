from __future__ import annotations

from inference_lab.models import JobStatus
from inference_lab.queue import JobQueue


def test_enqueue_and_result(queue: JobQueue) -> None:
    job_id = queue.enqueue("hello world", max_tokens=16)
    result = queue.get_result(job_id)
    assert result is not None
    assert result.status == JobStatus.queued
    assert result.prompt == "hello world"
    assert queue.stream_length() == 1


def test_consumer_ack_reduces_lag(queue: JobQueue) -> None:
    queue.ensure_group()
    for i in range(3):
        queue.enqueue(f"p{i}")
    assert queue.lag() == 3
    batch = queue.read_batch("worker-1", count=2)
    assert len(batch) == 2
    # 2 pending (unacked) + 1 never-delivered
    assert queue.lag() == 3
    for msg_id, _ in batch:
        queue.ack(msg_id)
    assert queue.lag() == 1
    batch2 = queue.read_batch("worker-1", count=5)
    assert len(batch2) == 1
    queue.ack(batch2[0][0])
    assert queue.lag() == 0


def test_mark_running_and_complete(queue: JobQueue) -> None:
    job_id = queue.enqueue("x")
    queue.mark_running(job_id, "worker-9")
    mid = queue.get_result(job_id)
    assert mid is not None and mid.status == JobStatus.running
    assert mid.worker_id == "worker-9"
