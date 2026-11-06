from __future__ import annotations

import pytest

from inference_lab.config import Settings
from inference_lab.memory_redis import MemoryRedis
from inference_lab.queue import JobQueue


@pytest.fixture
def settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/15",
        inference_mode="MOCK",
        mock_latency_ms=5,
        mock_latency_jitter_ms=0,
        scale_min=1,
        scale_max=4,
        scale_up_lag=5,
        scale_down_lag=1,
        hysteresis_hold_ticks=2,
        block_ms=10,
    )


@pytest.fixture
def mem() -> MemoryRedis:
    return MemoryRedis()


@pytest.fixture
def queue(settings: Settings, mem: MemoryRedis) -> JobQueue:
    return JobQueue(settings, client=mem)  # type: ignore[arg-type]
