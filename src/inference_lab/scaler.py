"""Autoscaling simulation: map queue lag → desired worker replicas with hysteresis."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass

from inference_lab.config import Settings, get_settings
from inference_lab.metrics import QUEUE_DEPTH, SCALE_DESIRED, SCALE_LAG, start_metrics_server
from inference_lab.models import ScaleState
from inference_lab.queue import JobQueue

log = logging.getLogger("inference_lab.scaler")


@dataclass
class ScaleDecision:
    desired: int
    lag: int
    in_flight: int
    reason: str


class Autoscaler:
    """
    Decision loop (not real K8s):
      - scale UP when lag >= scale_up_lag (held for hysteresis_hold_ticks)
      - scale DOWN when lag <= scale_down_lag (held for hysteresis_hold_ticks)
      - clamp to [scale_min, scale_max]
    Writes desired replicas to Redis so workers can idle when ordinal >= desired.
    """

    def __init__(self, settings: Settings | None = None, queue: JobQueue | None = None) -> None:
        self.settings = settings or get_settings()
        self.queue = queue or JobQueue(self.settings)
        self.desired = self.settings.scale_min
        self._up_ticks = 0
        self._down_ticks = 0
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def decide(self, lag: int, in_flight: int = 0) -> ScaleDecision:
        s = self.settings
        reason = "hold"
        next_desired = self.desired

        if lag >= s.scale_up_lag or in_flight >= s.scale_up_in_flight:
            self._up_ticks += 1
            self._down_ticks = 0
            if self._up_ticks >= s.hysteresis_hold_ticks and self.desired < s.scale_max:
                next_desired = min(s.scale_max, self.desired + 1)
                reason = f"scale_up lag={lag} in_flight={in_flight}"
                self._up_ticks = 0
            else:
                reason = f"scale_up_pending ticks={self._up_ticks}/{s.hysteresis_hold_ticks}"
        elif lag <= s.scale_down_lag and in_flight == 0:
            self._down_ticks += 1
            self._up_ticks = 0
            if self._down_ticks >= s.hysteresis_hold_ticks and self.desired > s.scale_min:
                next_desired = max(s.scale_min, self.desired - 1)
                reason = f"scale_down lag={lag}"
                self._down_ticks = 0
            else:
                reason = f"scale_down_pending ticks={self._down_ticks}/{s.hysteresis_hold_ticks}"
        else:
            self._up_ticks = 0
            self._down_ticks = 0
            reason = f"stable lag={lag} in_flight={in_flight}"

        self.desired = next_desired
        return ScaleDecision(desired=self.desired, lag=lag, in_flight=in_flight, reason=reason)

    def tick(self) -> ScaleState:
        lag = self.queue.lag()
        # Approximate in-flight as pending entries that are claimed but not acked
        decision = self.decide(lag=lag, in_flight=0)
        self.queue.set_desired_replicas(decision.desired)
        QUEUE_DEPTH.set(lag)
        SCALE_LAG.set(lag)
        SCALE_DESIRED.set(decision.desired)
        state = ScaleState(
            desired=decision.desired,
            lag=decision.lag,
            in_flight=decision.in_flight,
            reason=decision.reason,
            updated_at=time.time(),
        )
        log.info(
            "scaler desired=%s lag=%s reason=%s",
            state.desired,
            state.lag,
            state.reason,
        )
        return state

    def run_forever(self) -> None:
        self.queue.ensure_group()
        self.queue.set_desired_replicas(self.desired)
        log.info(
            "scaler start min=%s max=%s up_lag=%s down_lag=%s",
            self.settings.scale_min,
            self.settings.scale_max,
            self.settings.scale_up_lag,
            self.settings.scale_down_lag,
        )
        while not self._stop:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("scaler tick failed")
            time.sleep(self.settings.scale_interval_seconds)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    try:
        start_metrics_server(settings.metrics_port + 1)
    except OSError:
        log.warning("metrics port busy — skipping")
    scaler = Autoscaler(settings)

    def _handle(signum, frame):  # noqa: ANN001, ARG001
        scaler.request_stop()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    scaler.run_forever()


if __name__ == "__main__":
    run()
