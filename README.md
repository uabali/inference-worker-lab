# inference-worker-lab

Distributed Systems lab (3/5): **queue-decoupled inference** with a mock/vLLM worker pool and an **autoscaling simulation** driven by queue lag.

Target role signal: fal Software Engineer — Distributed Systems (backpressure, consumer groups, scale-on-lag).

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
     worker-1      worker-2      worker-N     (MOCK latency sim | VLLM HTTP)
          │             │             │
          └─────────────┴─────────────┘
                        │ SET result keys
                        ▼
                 ┌─────────────┐
                 │   scaler    │  lag → desired replicas (hysteresis)
                 └─────────────┘  workers with ordinal >= desired idle
```

**Modes**

| Mode | When | Behavior |
|------|------|----------|
| `MOCK` (default) | Mac / CI / no GPU | Fixed latency + token echo — no model download |
| `VLLM` | GPU box + compose profile `gpu` | OpenAI-compatible `POST /v1/chat/completions` |

## What this teaches

1. **Queue decoupling** — API returns immediately; workers pull via Redis Streams consumer group.
2. **Backpressure** — lag is visible (`/v1/scale`, Prometheus `iwl_queue_depth`); overload shows up as depth, not API timeouts.
3. **Autoscaling on lag** — scaler writes `desired` replicas with **hysteresis** (need N consecutive ticks before up/down) to avoid flapping.
4. **Partial capacity** — workers whose ordinal is above `desired` idle (simulates scale-in without real K8s).

## Quick start (no Docker daemon required for tests)

```bash
cd /Users/apple/Code/inference-worker-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Run locally with Redis

If Docker Desktop is up:

```bash
docker compose up --build
# API: http://127.0.0.1:8080
./scripts/demo_local.sh
```

Without compose, start Redis yourself, then three processes:

```bash
export IWL_REDIS_URL=redis://127.0.0.1:6379/0
export IWL_INFERENCE_MODE=MOCK

inference-api          # terminal 1
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

## Connect real vLLM (optional, GPU)

Compose does **not** pull the vLLM image unless you opt in:

```bash
# WARNING: large image + needs NVIDIA Container Toolkit
export IWL_INFERENCE_MODE=VLLM
export VLLM_MODEL=facebook/opt-125m   # tiny smoke model
docker compose --profile gpu up --build
```

Workers read `IWL_VLLM_BASE_URL` (default `http://vllm:8000/v1`). Any OpenAI-compatible server works (vLLM, TGI proxy, local llama.cpp server).

Env knobs (prefix `IWL_`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `INFERENCE_MODE` | `MOCK` | `MOCK` or `VLLM` |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI base URL |
| `SCALE_UP_LAG` | `10` | lag threshold to scale up |
| `SCALE_DOWN_LAG` | `2` | lag threshold to scale down |
| `SCALE_MIN` / `SCALE_MAX` | `1` / `8` | replica clamps |
| `HYSTERESIS_HOLD_TICKS` | `2` | consecutive ticks before change |

## Metrics

- API: `GET /metrics` (Prometheus)
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
  memory_redis.py # test double (no daemon)
tests/            # pytest, Redis-free
docker-compose.yml
```

## Demo checklist for review

1. `pytest -q` green without Docker.
2. Burst 12 jobs → `/v1/scale` shows rising `desired` after hysteresis.
3. Flip `IWL_INFERENCE_MODE=VLLM` only when a real endpoint exists — never call paid APIs from this lab.

## Related labs

1. [rust-queue-worker](https://github.com/uabali/rust-queue-worker) — Tokio queue + retry/backoff  
2. [mini-CDN](https://github.com/uabali/mini-CDN) — consistent hashing, load shedding  
3. **inference-worker-lab** (this repo)  
4. Linux/networking lab (next)  
5. DDIA + K8s/GPU scheduling reading notes (next)
