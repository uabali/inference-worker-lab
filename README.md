# inference-worker-lab

Queue-decoupled inference: FastAPI gateway, Redis Streams workers (mock or vLLM), idle-claim recovery, bounded retries, and a lag/pending autoscaler.

## Architecture

```
                 ┌─────────────┐
  client ──────► │  API (:8080)│  POST /v1/jobs  → job_id
                 │  FastAPI    │  GET  /v1/jobs/{id}
                 └──────┬──────┘  GET  /v1/scale
                        │ XADD
                        ▼
                 ┌─────────────┐
                 │ Redis Stream│  inference:jobs  (consumer group: workers)
                 └──────┬──────┘
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     worker-1      worker-2      worker-N     (MOCK | VLLM HTTP)
          │             │             │
          └─────────────┴─────────────┘
                        │ SET result keys
                        ▼
                 ┌─────────────┐
                 │   scaler    │  never-delivered lag + PEL pending
                 └─────────────┘  hysteresis; ordinal >= desired idles
```

Workers `XAUTOCLAIM` pending messages idle for `IWL_CLAIM_IDLE_MS` (default 30s). A crash after `XREADGROUP` and before `XACK` is picked up by another worker. If the result key is already `succeeded`, the reclaim acks and does not call the model again.

A failed attempt is `XADD`ed again with `attempt+1`, then the old id is acked. At `IWL_MAX_ATTEMPTS` the job is marked failed and appended to `inference:dlq`. A crash between that `XADD` and `XACK` can leave two copies; both may run. That is at-least-once, not exactly-once.

`POST /v1/jobs` accepts `idempotency_key`. The same key returns the same `job_id` and does not append a second stream entry. There is a crash window after the key is stored and before `XADD`.

Enqueue returns 429 when unfinished work (`lag`) is at `IWL_MAX_QUEUE_DEPTH` (default 10000, `0` disables).

| Mode | When | Behavior |
|------|------|----------|
| `MOCK` (default) | Mac / CI / no GPU | Fixed latency + token echo |
| `VLLM` | GPU + compose profile `gpu` | OpenAI-compatible `POST /v1/chat/completions` |

## Test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

`tests/test_redis_integration.py` uses Redis at `IWL_TEST_REDIS_URL` (default `redis://127.0.0.1:6399/0`) and skips if nothing is listening. The other tests use `MemoryRedis` and do not need a daemon. vLLM and the compose `gpu` profile are not part of `pytest`.

## Run with Docker

```bash
docker compose up --build
# API: http://127.0.0.1:8080
./scripts/demo_local.sh
```

## Run without Compose

Start Redis, then:

```bash
export IWL_REDIS_URL=redis://127.0.0.1:6379/0
export IWL_INFERENCE_MODE=MOCK

inference-api
IWL_WORKER_ID=worker-1 inference-worker
IWL_WORKER_ID=worker-2 inference-worker
inference-scaler
```

Enqueue:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"explain consistent hashing","max_tokens":32}'
curl -s http://127.0.0.1:8080/v1/jobs/<job_id>
curl -s http://127.0.0.1:8080/v1/scale
```

## vLLM (optional, GPU)

```bash
export IWL_INFERENCE_MODE=VLLM
export VLLM_MODEL=facebook/opt-125m
docker compose --profile gpu up --build
```

Workers use `IWL_VLLM_BASE_URL` (default `http://vllm:8000/v1`). Any OpenAI-compatible server works.

## Config (`IWL_` prefix)

| Variable | Default | Meaning |
|----------|---------|---------|
| `INFERENCE_MODE` | `MOCK` | `MOCK` or `VLLM` |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI base URL |
| `SCALE_UP_LAG` | `10` | never-delivered entries before scale-up |
| `SCALE_DOWN_LAG` | `2` | scale-down only when lag is at or below this and pending is 0 |
| `SCALE_UP_IN_FLIGHT` | `4` | claimed-but-unacked entries can also scale up |
| `SCALE_MIN` / `SCALE_MAX` | `1` / `8` | replica clamps |
| `HYSTERESIS_HOLD_TICKS` | `2` | consecutive ticks before a change |
| `CLAIM_IDLE_MS` | `30000` | min idle before `XAUTOCLAIM` |
| `MAX_ATTEMPTS` | `3` | tries before the dead-letter stream |
| `MAX_QUEUE_DEPTH` | `10000` | unfinished jobs before HTTP 429; `0` disables |

The scaler does not start or stop containers. It writes a desired count. A worker whose id is `worker-N` idles when `N-1 >= desired`. Unknown ids stay active. Compose starts a fixed set of worker processes.

## Metrics

- `GET /metrics` (Prometheus)
- Gauges: `iwl_queue_depth` (never-delivered + pending), `iwl_in_flight`, `iwl_scale_desired`, `iwl_scale_lag` (never-delivered only), `iwl_scale_pending`
- Counters: `iwl_jobs_total`, `iwl_retries_total`, `iwl_dlq_total`, `iwl_reclaims_total`
- Histogram: `iwl_inference_latency_seconds` is backend `generate()` time, not queue wait

## Sample measurement

`scripts/measure_mock.py` runs 80 mock jobs in one process against `MemoryRedis` (concurrency 4, mock sleep 5 ms, jitter 0). Nearest-rank percentiles. Not a GPU, Redis, or production result.

Two consecutive runs on this host (`Linux-6.12.113+`, 4 vCPU, `Intel(R) Xeon(R) Processor`, no GPU), 2026-11-08:

| run | wall_s | succeeded | backend p50/p95/p99 ms | queue_to_done p50/p95/p99 ms |
|-----|--------|-----------|------------------------|------------------------------|
| 1 | 0.2277 | 80/80 | 5.00 / 5.00 / 5.00 | 121.64 / 212.86 / 224.39 |
| 2 | 0.2273 | 80/80 | 5.00 / 5.00 / 5.00 | 123.62 / 215.71 / 227.06 |

Backend latency matches the configured 5 ms sleep. `queue_to_done` is `finished_at - created_at` for that in-process loop.

Docker Compose and the vLLM profile were not started for these runs. A real Redis 7.0.15 on `127.0.0.1:6399` was used only by `tests/test_redis_integration.py` (2 passed): idle pending is not reclaimed before `min_idle`, and a worker completes a message left unacked by a crashed consumer.

## Layout

```
src/inference_lab/
  api.py          # FastAPI gateway
  worker.py       # stream consumer + MOCK/VLLM
  scaler.py       # lag → desired replicas
  queue.py        # Redis Streams wrapper
  inference.py    # backends
  metrics.py
  memory_redis.py # test double
tests/
docker-compose.yml
```
