"""API gateway: enqueue prompts, poll job status/results, expose scale + health."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inference_lab.config import Settings, get_settings
from inference_lab.metrics import QUEUE_DEPTH
from inference_lab.models import EnqueueRequest, EnqueueResponse, JobResult, JobStatus
from inference_lab.queue import JobQueue, QueueFull

log = logging.getLogger("inference_lab.api")


def create_app(settings: Settings | None = None, queue: JobQueue | None = None) -> FastAPI:
    settings = settings or get_settings()
    q = queue or JobQueue(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            q.ensure_group()
        except Exception as exc:  # noqa: BLE001
            log.warning("redis not ready at startup: %s (API still up for /healthz)", exc)
        app.state.queue = q
        app.state.settings = settings
        yield

    app = FastAPI(title="inference-worker-lab", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/jobs", response_model=EnqueueResponse)
    def enqueue(body: EnqueueRequest) -> EnqueueResponse:
        try:
            job_id = q.enqueue(
                body.prompt,
                body.max_tokens,
                body.metadata,
                idempotency_key=body.idempotency_key,
            )
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}") from exc
        try:
            QUEUE_DEPTH.set(q.lag())
        except Exception:  # noqa: BLE001
            pass
        return EnqueueResponse(job_id=job_id, status=JobStatus.queued)

    @app.get("/v1/jobs/{job_id}", response_model=JobResult)
    def get_job(job_id: str) -> JobResult:
        result = q.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="job not found")
        return result

    @app.get("/v1/scale")
    def get_scale() -> dict[str, int | str]:
        try:
            never_delivered, pending = q.backlog()
            desired = q.get_desired_replicas(default=settings.scale_min)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "desired": desired,
            "lag": never_delivered + pending,
            "never_delivered": never_delivered,
            "pending": pending,
            "stream_length": q.stream_length(),
        }

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    uvicorn.run(
        "inference_lab.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
