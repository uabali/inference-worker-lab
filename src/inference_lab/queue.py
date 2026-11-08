"""Redis Streams job queue with consumer group, reclaim, and a dead-letter stream."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from inference_lab.config import Settings
from inference_lab.models import JobResult, JobStatus


class QueueFull(Exception):
    """Raised when unfinished work is already at `max_queue_depth`."""

    def __init__(self, depth: int, limit: int) -> None:
        super().__init__(f"queue depth {depth} >= {limit}")
        self.depth = depth
        self.limit = limit


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

    def enqueue(
        self,
        prompt: str,
        max_tokens: int = 64,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        idem = (idempotency_key or "").strip() or None
        if idem:
            existing = self.r.get(self._idem_key(idem))
            if existing:
                return str(existing)

        if self.settings.max_queue_depth > 0:
            depth = self.lag()
            if depth >= self.settings.max_queue_depth:
                raise QueueFull(depth, self.settings.max_queue_depth)

        job_id = str(uuid.uuid4())
        if idem:
            stored = self.r.set(
                self._idem_key(idem),
                job_id,
                nx=True,
                ex=self.settings.result_ttl_seconds,
            )
            if not stored:
                existing = self.r.get(self._idem_key(idem))
                if existing:
                    return str(existing)

        created_at = str(time.time())
        payload = {
            "job_id": job_id,
            "prompt": prompt,
            "max_tokens": str(max_tokens),
            "metadata": json.dumps(metadata or {}),
            "created_at": created_at,
            "attempt": "0",
        }
        self.r.xadd(self.settings.stream_key, payload)
        self._store_result(
            JobResult(
                job_id=job_id,
                status=JobStatus.queued,
                prompt=prompt,
                created_at=float(created_at),
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
        return _messages(resp)

    def reclaim(
        self,
        consumer: str,
        count: int,
        min_idle_ms: int | None = None,
    ) -> list[tuple[str, dict[str, str]]]:
        """Take pending messages idle at least `min_idle_ms` (default: settings.claim_idle_ms).

        Redis assigns them to `consumer`. A message that was read and never acked
        (worker crash) becomes visible again. min idle 0 claims every pending entry;
        the running worker uses claim_idle_ms so it does not steal its own fresh work.
        """
        self.ensure_group()
        idle = self.settings.claim_idle_ms if min_idle_ms is None else min_idle_ms
        try:
            resp = self.r.xautoclaim(
                self.settings.stream_key,
                self.settings.consumer_group,
                consumer,
                idle,
                start_id="0-0",
                count=count,
            )
        except redis.ResponseError:
            return []
        return _claimed(resp)

    def requeue(self, fields: dict[str, str]) -> str:
        """Append another attempt of the same job. Caller acks the old id after this returns."""
        attempt = int(fields.get("attempt", "0")) + 1
        payload = {
            "job_id": fields.get("job_id", ""),
            "prompt": fields.get("prompt", ""),
            "max_tokens": fields.get("max_tokens", "64"),
            "metadata": fields.get("metadata", "{}"),
            "created_at": fields.get("created_at", str(time.time())),
            "attempt": str(attempt),
        }
        return str(self.r.xadd(self.settings.stream_key, payload))

    def dead_letter(self, fields: dict[str, str], error: str) -> str:
        payload = {
            "job_id": fields.get("job_id", ""),
            "prompt": fields.get("prompt", ""),
            "attempt": fields.get("attempt", "0"),
            "error": error[:500],
            "dead_lettered_at": str(time.time()),
        }
        return str(self.r.xadd(self.settings.dlq_stream_key, payload))

    def ack(self, msg_id: str) -> None:
        self.r.xack(self.settings.stream_key, self.settings.consumer_group, msg_id)

    def backlog(self) -> tuple[int, int]:
        """Return (never_delivered, pending).

        never_delivered is Redis group lag (not yet read). pending is the PEL
        (read, not acked), which includes in-flight work and crashed claims.
        """
        self.ensure_group()
        pending = 0
        try:
            info = self.r.xpending(self.settings.stream_key, self.settings.consumer_group)
            pending = int(info.get("pending", 0) if isinstance(info, dict) else info[0])
        except redis.ResponseError:
            pending = 0
        never_delivered = 0
        try:
            groups = self.r.xinfo_groups(self.settings.stream_key)
            for group in groups:
                if group.get("name") == self.settings.consumer_group and group.get("lag") is not None:
                    never_delivered = int(group["lag"])
                    break
        except redis.ResponseError:
            pass
        return never_delivered, pending

    def lag(self) -> int:
        """Unfinished work: never delivered + pending (claimed, not acked)."""
        never_delivered, pending = self.backlog()
        return never_delivered + pending

    def stream_length(self) -> int:
        try:
            return int(self.r.xlen(self.settings.stream_key))
        except redis.ResponseError:
            return 0

    def _idem_key(self, key: str) -> str:
        return f"{self.settings.idempotency_prefix}{key}"

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


def _messages(resp: Any) -> list[tuple[str, dict[str, str]]]:
    out: list[tuple[str, dict[str, str]]] = []
    if not resp:
        return out
    for _stream, messages in resp:
        for msg_id, fields in messages:
            out.append((str(msg_id), {str(k): str(v) for k, v in fields.items()}))
    return out


def _claimed(resp: Any) -> list[tuple[str, dict[str, str]]]:
    # redis-py: [next_start_id, [(id, fields), ...], [deleted_ids]]
    if not resp or not isinstance(resp, (list, tuple)) or len(resp) < 2:
        return []
    out: list[tuple[str, dict[str, str]]] = []
    for item in resp[1] or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
            out.append((str(item[0]), {str(k): str(v) for k, v in item[1].items()}))
    return out
