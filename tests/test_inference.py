from __future__ import annotations

import pytest

from inference_lab.config import Settings
from inference_lab.inference import MockBackend, build_backend


@pytest.mark.asyncio
async def test_mock_backend_returns_tokens(settings: Settings) -> None:
    backend = MockBackend(settings)
    out = await backend.generate("ping", max_tokens=8)
    assert out.mode == "MOCK"
    assert out.tokens == 8
    assert "ping" in out.text
    assert out.latency_ms >= settings.mock_latency_ms


def test_build_backend_default_mock(settings: Settings) -> None:
    assert build_backend(settings).__class__.__name__ == "MockBackend"
