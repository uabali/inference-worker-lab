"""Job and result models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class EnqueueRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16_000)
    max_tokens: int = Field(default=64, ge=1, le=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnqueueResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    prompt: str = ""
    output: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    tokens: int | None = None
    worker_id: str | None = None
    mode: str | None = None
    created_at: float | None = None
    finished_at: float | None = None


class ScaleState(BaseModel):
    desired: int
    lag: int
    in_flight: int
    reason: str
    updated_at: float
