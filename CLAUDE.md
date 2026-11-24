# CLAUDE.md

Redis Streams inference queue: FastAPI gateway, workers (MOCK or vLLM backend), idle-claim recovery, bounded retries with a DLQ, and a lag/pending autoscaler that only writes a desired replica count. Python >= 3.11.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest -q                      # MemoryRedis tests; Redis integration test skips without a daemon
ruff check .                   # config in pyproject.toml (E/F/W)
docker compose up --build      # redis + api + 2 mock workers + scaler
inference-api | inference-worker | inference-scaler   # console scripts
```

`tests/test_redis_integration.py` needs Redis at `IWL_TEST_REDIS_URL` (default `redis://127.0.0.1:6399/0`).

## Layout

- `src/inference_lab/api.py` gateway (`/v1/jobs`, `/v1/scale`, `/metrics`), `worker.py` consumer, `scaler.py`, `queue.py` Redis Streams wrapper, `inference.py` backends, `config.py` settings (`IWL_` env prefix), `metrics.py`, `memory_redis.py` in-memory test double.
- `tests/` pytest (asyncio_mode=auto). `scripts/` demo and mock measurement.

## Conventions and gotchas

- Delivery is at-least-once, not exactly-once; do not claim otherwise in docs.
- Config lives only in `Settings`; document new variables in README and `.env.example`.
- `MemoryRedis` must mirror the Redis commands `queue.py` uses; extend it when adding commands.
- The scaler never starts containers; workers named `worker-N` idle when `N-1 >= desired`.
- `requirements.txt` is pinned; `pyproject.toml` holds loose lower bounds.
