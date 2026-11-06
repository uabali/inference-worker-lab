from __future__ import annotations

from inference_lab.config import Settings
from inference_lab.queue import JobQueue
from inference_lab.scaler import Autoscaler


def test_scale_up_with_hysteresis(settings: Settings, queue: JobQueue) -> None:
    scaler = Autoscaler(settings, queue=queue)
    d1 = scaler.decide(lag=10, in_flight=0)
    assert d1.desired == 1
    assert "pending" in d1.reason
    d2 = scaler.decide(lag=10, in_flight=0)
    assert d2.desired == 2
    assert "scale_up" in d2.reason


def test_scale_down_with_hysteresis(settings: Settings, queue: JobQueue) -> None:
    scaler = Autoscaler(settings, queue=queue)
    scaler.desired = 3
    d1 = scaler.decide(lag=0, in_flight=0)
    assert d1.desired == 3
    d2 = scaler.decide(lag=0, in_flight=0)
    assert d2.desired == 2


def test_clamp_to_max(settings: Settings, queue: JobQueue) -> None:
    scaler = Autoscaler(settings, queue=queue)
    scaler.desired = settings.scale_max
    d = scaler.decide(lag=100, in_flight=0)
    for _ in range(4):
        d = scaler.decide(lag=100, in_flight=0)
    assert d.desired == settings.scale_max


def test_scaler_writes_desired(queue: JobQueue, settings: Settings) -> None:
    scaler = Autoscaler(settings, queue=queue)
    for i in range(6):
        queue.enqueue(f"job-{i}")
    s1 = scaler.tick()
    assert s1.desired == 1
    s2 = scaler.tick()
    assert s2.desired == 2
    assert queue.get_desired_replicas() == 2
