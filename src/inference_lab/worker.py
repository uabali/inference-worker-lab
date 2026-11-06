"""Inference worker: pull from Redis Streams, run MOCK/VLLM, write results."""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from inference_lab.config import Settings, get_settings
from inference_lab.inference import InferenceBackend, build_backend
from inference_lab.metrics import IN_FLIGHT, INFERENCE_LATENCY, JOBS_TOTAL, start_metrics_server
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

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        self.queue.ensure_group()
        log.info(
            "worker start id=%s mode=%s concurrency=%s",
            self.settings.worker_id,
            self.settings.inference_mode,
            self.settings.worker_concurrency,
        )
        sem = asyncio.Semaphore(self.settings.worker_concurrency)
        tasks: set[asyncio.Task] = set()

        while not self._stop.is_set():
            # Respect scaler desired replicas: if this worker's ordinal is above
            # desired, idle instead of draining the queue (simulates scale-down).
            desired = self.queue.get_desired_replicas(default=self.settings.scale_max)
            ordinal = _worker_ordinal(self.settings.worker_id)
            if ordinal >= desired:
                await asyncio.sleep(0.5)
                continue

            batch = await asyncio.to_thread(
                self.queue.read_batch,
                self.settings.worker_id,
                self.settings.worker_concurrency,
            )
            if not batch:
                await asyncio.sleep(0.05)
                continue

            for msg_id, fields in batch:
                await sem.acquire()
                task = asyncio.create_task(self._handle(msg_id, fields, sem))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle(self, msg_id: str, fields: dict[str, str], sem: asyncio.Semaphore) -> None:
        job_id = fields.get("job_id", msg_id)
        prompt = fields.get("prompt", "")
        max_tokens = int(fields.get("max_tokens", "64"))
        created_at = float(fields.get("created_at", time.time()))
        self._in_flight += 1
        IN_FLIGHT.set(self._in_flight)
        self.queue.mark_running(job_id, self.settings.worker_id)
        started = time.perf_counter()
        try:
            out = await self.backend.generate(prompt, max_tokens)
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
            self.queue.ack(msg_id)
            JOBS_TOTAL.labels(status="succeeded", mode=out.mode).inc()
            log.info("job ok id=%s latency_ms=%.1f tokens=%s", job_id, out.latency_ms, out.tokens)
        except Exception as exc:  # noqa: BLE001 — surface to job result
            JOBS_TOTAL.labels(status="failed", mode=self.settings.inference_mode.upper()).inc()
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
            self.queue.ack(msg_id)
            log.exception("job failed id=%s err=%s", job_id, exc)
        finally:
            self._in_flight = max(0, self._in_flight - 1)
            IN_FLIGHT.set(self._in_flight)
            sem.release()


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
