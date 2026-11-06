"""Redis Streams job queue with consumer group."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from inference_lab.config import Settings
from inference_lab.models import JobResult, JobStatus


class JobQueue:
    """Thin Redis Streams wrapper used by API, workers, and scaler."""

    def __init__(self, settings: Settings, client: redis.Redis | None = None) -> None:
        self.settings = settings
        self.r = client or redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(
                self.settings.stream_key,
                self.settings.consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, prompt: str, max_tokens: int = 64, metadata: dict[str, Any] | None = None) -> str:
        job_id = str(uuid.uuid4())
        payload = {
            "job_id": job_id,
            "prompt": prompt,
            "max_tokens": str(max_tokens),
            "metadata": json.dumps(metadata or {}),
            "created_at": str(time.time()),
        }
        self.r.xadd(self.settings.stream_key, payload)
        self._store_result(
            JobResult(
                job_id=job_id,
                status=JobStatus.queued,
                prompt=prompt,
                created_at=float(payload["created_at"]),
            )
        )
        return job_id

    def read_batch(self, consumer: str, count: int = 1) -> list[tuple[str, dict[str, str]]]:
        """Claim new messages for this consumer. Returns [(msg_id, fields), ...]."""
        self.ensure_group()
        resp = self.r.xreadgroup(
            groupname=self.settings.consumer_group,
            consumername=consumer,
            streams={self.settings.stream_key: ">"},
            count=count,
            block=self.settings.block_ms,
        )
        out: list[tuple[str, dict[str, str]]] = []
        if not resp:
            return out
        for _stream, messages in resp:
            for msg_id, fields in messages:
                out.append((msg_id, fields))
        return out

    def ack(self, msg_id: str) -> None:
        self.r.xack(self.settings.stream_key, self.settings.consumer_group, msg_id)

    def lag(self) -> int:
        """Queue depth ≈ never-delivered (Redis 7 lag) + pending (claimed, unacked)."""
        self.ensure_group()
        try:
            info = self.r.xpending(self.settings.stream_key, self.settings.consumer_group)
            pending = int(info.get("pending", 0) if isinstance(info, dict) else info[0])
        except redis.ResponseError:
            pending = 0
        never_delivered = 0
        try:
            groups = self.r.xinfo_groups(self.settings.stream_key)
            for g in groups:
                if g.get("name") == self.settings.consumer_group:
                    if "lag" in g and g["lag"] is not None:
                        never_delivered = int(g["lag"])
                    break
        except redis.ResponseError:
            pass
        return pending + never_delivered

    def stream_length(self) -> int:
        try:
            return int(self.r.xlen(self.settings.stream_key))
        except redis.ResponseError:
            return 0

    def _result_key(self, job_id: str) -> str:
        return f"{self.settings.result_prefix}{job_id}"

    def _store_result(self, result: JobResult) -> None:
        self.r.set(
            self._result_key(result.job_id),
            result.model_dump_json(),
            ex=self.settings.result_ttl_seconds,
        )

    def mark_running(self, job_id: str, worker_id: str) -> None:
        current = self.get_result(job_id)
        if current is None:
            current = JobResult(job_id=job_id, status=JobStatus.running)
        current.status = JobStatus.running
        current.worker_id = worker_id
        self._store_result(current)

    def complete(self, result: JobResult) -> None:
        self._store_result(result)

    def get_result(self, job_id: str) -> JobResult | None:
        raw = self.r.get(self._result_key(job_id))
        if not raw:
            return None
        return JobResult.model_validate_json(raw)

    def set_desired_replicas(self, desired: int) -> None:
        self.r.set(self.settings.scale_state_key, str(desired))

    def get_desired_replicas(self, default: int = 1) -> int:
        raw = self.r.get(self.settings.scale_state_key)
        if raw is None:
            return default
        return int(raw)
