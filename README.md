# inference-worker-lab

Queue-decoupled inference: FastAPI gateway, Redis Streams worker pool (mock or vLLM), and lag-based autoscaling with hysteresis.

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
                 │   scaler    │  lag → desired replicas (hysteresis)
                 └─────────────┘  workers with ordinal >= desired idle
```

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
| `SCALE_UP_LAG` | `10` | scale-up lag threshold |
| `SCALE_DOWN_LAG` | `2` | scale-down lag threshold |
| `SCALE_MIN` / `SCALE_MAX` | `1` / `8` | replica clamps |
| `HYSTERESIS_HOLD_TICKS` | `2` | consecutive ticks before change |

## Metrics

- `GET /metrics` (Prometheus)
- Gauges: `iwl_queue_depth`, `iwl_in_flight`, `iwl_scale_desired`, `iwl_scale_lag`
- Histogram: `iwl_inference_latency_seconds`

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
