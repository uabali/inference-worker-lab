"""Runtime configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IWL_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    stream_key: str = "inference:jobs"
    consumer_group: str = "workers"
    result_prefix: str = "inference:result:"
    result_ttl_seconds: int = 3600
    dlq_stream_key: str = "inference:dlq"
    idempotency_prefix: str = "inference:idem:"
    # 0 disables the enqueue cap. Depth is never-delivered + pending.
    max_queue_depth: int = 10_000
    max_attempts: int = 3

    # MOCK | VLLM
    inference_mode: str = "MOCK"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    vllm_timeout_seconds: float = 120.0

    mock_latency_ms: int = 80
    mock_latency_jitter_ms: int = 40
    mock_tokens_per_prompt: int = 32

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    metrics_port: int = 9100

    # Worker
    worker_id: str = "worker-1"
    worker_concurrency: int = 2
    # XAUTOCLAIM min idle. Must be longer than a normal attempt so a live worker
    # is not stolen. Crashed claims become visible after this idle time.
    claim_idle_ms: int = 30_000
    block_ms: int = 2_000

    # Scaler
    scale_min: int = 1
    scale_max: int = 8
    scale_up_lag: int = 10
    scale_down_lag: int = 2
    scale_up_in_flight: int = 4
    scale_interval_seconds: float = 2.0
    scale_state_key: str = "inference:scaler:desired"
    hysteresis_hold_ticks: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
