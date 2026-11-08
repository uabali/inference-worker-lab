"""Inference worker: pull from Redis Streams, reclaim idle claims, run MOCK/VLLM."""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from inference_lab.config import Settings, get_settings
from inference_lab.inference import InferenceBackend, build_backend
from inference_lab.metrics import (
    DLQ_TOTAL,
    IN_FLIGHT,
    INFERENCE_LATENCY,
    JOBS_TOTAL,
    RECLAIMS,
    RETRIES,
    start_metrics_server,
)
from inference_lab.models import JobResult, JobStatus
from inference_lab.queue import JobQueue

log = logging.getLogger("inference_lab.worker")


class Worker:
    def __init__(
        self,
        settings: Settings | None = None,
        queue: JobQueue | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.queue = queue or JobQueue(self.settings)
        self.backend = backend or build_backend(self.settings)
        self._stop = asyncio.Event()
        self._in_flight = 0
        self._reserved = 0
        self._inflight_ids: set[str] = set()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        self.queue.ensure_group()
        log.info(
            "worker start id=%s mode=%s concurrency=%s claim_idle_ms=%s max_attempts=%s",
            self.settings.worker_id,
            self.settings.inference_mode,
            self.settings.worker_concurrency,
            self.settings.claim_idle_ms,
            self.settings.max_attempts,
        )
        sem = asyncio.Semaphore(self.settings.worker_concurrency)
        tasks: set[asyncio.Task] = set()

        while not self._stop.is_set():
            # Above the scaler's desired count: do not take new or idle work.
            # In-flight tasks started earlier still finish (see gather below).
            desired = self.queue.get_desired_replicas(default=self.settings.scale_max)
            ordinal = _worker_ordinal(self.settings.worker_id)
            if ordinal >= desired:
                await asyncio.sleep(0.05)
                continue

            slots = self.settings.worker_concurrency - self._reserved
            if slots <= 0:
                await asyncio.sleep(0.01)
                continue

            reclaimed = await asyncio.to_thread(
                self.queue.reclaim,
                self.settings.worker_id,
                slots,
                self.settings.claim_idle_ms,
            )
            if reclaimed:
                RECLAIMS.inc(len(reclaimed))
            remaining = slots - len(reclaimed)
            fresh: list[tuple[str, dict[str, str]]] = []
            if remaining > 0 and not self._stop.is_set():
                fresh = await asyncio.to_thread(
                    self.queue.read_batch,
                    self.settings.worker_id,
                    remaining,
                )
            batch = _dedupe(reclaimed + fresh)
            if not batch:
                await asyncio.sleep(0.02)
                continue

            scheduled = 0
            for msg_id, fields in batch:
                # This batch is already claimed. Finish it on the way out so a stop
                # does not leave fresh reads sitting in the PEL until claim_idle_ms.
                if msg_id in self._inflight_ids:
                    continue
                self._inflight_ids.add(msg_id)
                self._reserved += 1
                scheduled += 1
                await sem.acquire()
                task = asyncio.create_task(self._handle(msg_id, fields, sem))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            if scheduled == 0:
                # Reclaim handed back a message this process is already running.
                await asyncio.sleep(0.01)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle(self, msg_id: str, fields: dict[str, str], sem: asyncio.Semaphore) -> None:
        job_id = fields.get("job_id", msg_id)
        counted = False
        try:
            existing = self.queue.get_result(job_id)
            if existing is not None and existing.status == JobStatus.succeeded:
                # Crash after the result write and before ack. Do not run the model again.
                self.queue.ack(msg_id)
                return

            prompt = fields.get("prompt", "")
            max_tokens = int(fields.get("max_tokens", "64"))
            created_at = float(fields.get("created_at", time.time()))
            attempt = int(fields.get("attempt", "0"))
            self._in_flight += 1
            counted = True
            IN_FLIGHT.set(self._in_flight)
            self.queue.mark_running(job_id, self.settings.worker_id)
            started = time.perf_counter()
            try:
                out = await self.backend.generate(prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001 — recorded on the job, then retry or DLQ
                self._fail(msg_id, fields, job_id, prompt, created_at, attempt, exc)
                return

            latency_s = time.perf_counter() - started
            INFERENCE_LATENCY.observe(latency_s)
            result = JobResult(
                job_id=job_id,
                status=JobStatus.succeeded,
                prompt=prompt,
                output=out.text,
                latency_ms=out.latency_ms,
                tokens=out.tokens,
                worker_id=self.settings.worker_id,
                mode=out.mode,
                created_at=created_at,
                finished_at=time.time(),
            )
            self.queue.complete(result)
            try:
                self.queue.ack(msg_id)
            except Exception:
                # Result is already succeeded. Reclaim will see that and ack later.
                log.exception("ack failed after success id=%s", job_id)
            JOBS_TOTAL.labels(status="succeeded", mode=out.mode).inc()
            log.info("job ok id=%s latency_ms=%.1f tokens=%s", job_id, out.latency_ms, out.tokens)
        finally:
            self._inflight_ids.discard(msg_id)
            if counted:
                self._in_flight = max(0, self._in_flight - 1)
                IN_FLIGHT.set(self._in_flight)
            self._reserved = max(0, self._reserved - 1)
            sem.release()

    def _fail(
        self,
        msg_id: str,
        fields: dict[str, str],
        job_id: str,
        prompt: str,
        created_at: float,
        attempt: int,
        exc: Exception,
    ) -> None:
        log.warning("job attempt failed id=%s attempt=%s err=%s", job_id, attempt, exc)
        if attempt + 1 >= self.settings.max_attempts:
            result = JobResult(
                job_id=job_id,
                status=JobStatus.failed,
                prompt=prompt,
                error=str(exc),
                worker_id=self.settings.worker_id,
                mode=self.settings.inference_mode.upper(),
                created_at=created_at,
                finished_at=time.time(),
            )
            self.queue.complete(result)
            try:
                self.queue.dead_letter(fields, str(exc))
            except Exception:
                log.exception("dlq write failed id=%s; leaving pending", job_id)
                return
            self.queue.ack(msg_id)
            DLQ_TOTAL.inc()
            JOBS_TOTAL.labels(status="failed", mode=self.settings.inference_mode.upper()).inc()
            return

        try:
            self.queue.requeue(fields)
        except Exception:
            log.exception("requeue failed id=%s; leaving pending for reclaim", job_id)
            return
        self.queue.ack(msg_id)
        RETRIES.inc()


def _dedupe(batch: list[tuple[str, dict[str, str]]]) -> list[tuple[str, dict[str, str]]]:
    seen: set[str] = set()
    out: list[tuple[str, dict[str, str]]] = []
    for msg_id, fields in batch:
        if msg_id in seen:
            continue
        seen.add(msg_id)
        out.append((msg_id, fields))
    return out


def _worker_ordinal(worker_id: str) -> int:
    """worker-1 -> 0, worker-3 -> 2; unknown ids stay active (ordinal 0)."""
    if "-" in worker_id:
        tail = worker_id.rsplit("-", 1)[-1]
        if tail.isdigit():
            return max(0, int(tail) - 1)
    return 0


async def _amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    try:
        start_metrics_server(settings.metrics_port)
    except OSError:
        log.warning("metrics port %s busy — skipping", settings.metrics_port)
    worker = Worker(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.request_stop)
        except NotImplementedError:
            pass
    await worker.run_forever()


def run() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    run()
