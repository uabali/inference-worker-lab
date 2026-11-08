"""Prometheus metrics (optional — no-op scrape if unused)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

QUEUE_DEPTH = Gauge("iwl_queue_depth", "Approximate pending jobs in the stream")
IN_FLIGHT = Gauge("iwl_in_flight", "Jobs currently being processed by this process")
INFERENCE_LATENCY = Histogram(
    "iwl_inference_latency_seconds",
    "backend generate() seconds; does not include queue wait",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
JOBS_TOTAL = Counter("iwl_jobs_total", "Terminal job outcomes", ["status", "mode"])
RETRIES = Counter("iwl_retries_total", "Attempts requeued after a failure")
DLQ_TOTAL = Counter("iwl_dlq_total", "Jobs appended to the dead-letter stream")
RECLAIMS = Counter("iwl_reclaims_total", "Idle pending messages claimed with XAUTOCLAIM")
SCALE_DESIRED = Gauge("iwl_scale_desired", "Desired worker replicas from scaler")
SCALE_LAG = Gauge("iwl_scale_lag", "Never-delivered stream lag observed by the scaler")
SCALE_PENDING = Gauge("iwl_scale_pending", "Claimed-but-unacked entries observed by the scaler")


def start_metrics_server(port: int) -> None:
    start_http_server(port)
