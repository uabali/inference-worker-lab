"""Inference backends: MOCK (no GPU) and VLLM (OpenAI-compatible HTTP)."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from inference_lab.config import Settings


@dataclass
class InferenceOutput:
    text: str
    tokens: int
    latency_ms: float
    mode: str


class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int) -> InferenceOutput:
        raise NotImplementedError


class MockBackend(InferenceBackend):
    """Deterministic-ish latency + token simulation — default for Mac / CI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, prompt: str, max_tokens: int) -> InferenceOutput:
        base = self.settings.mock_latency_ms
        jitter = self.settings.mock_latency_jitter_ms
        delay_ms = base + random.randint(0, max(0, jitter))
        # Mild backpressure signal: longer prompts cost a bit more
        delay_ms += min(200, len(prompt) // 50)
        await asyncio.sleep(delay_ms / 1000.0)
        tokens = min(max_tokens, self.settings.mock_tokens_per_prompt)
        text = f"[mock] echoed ({tokens} tok): {prompt[:120]}"
        return InferenceOutput(text=text, tokens=tokens, latency_ms=float(delay_ms), mode="MOCK")


class VLLMBackend(InferenceBackend):
    """Calls an OpenAI-compatible /v1/chat/completions endpoint (vLLM / TGI proxy)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, prompt: str, max_tokens: int) -> InferenceOutput:
        url = f"{self.settings.vllm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.settings.vllm_timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        usage = data.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or max_tokens)
        return InferenceOutput(text=text, tokens=tokens, latency_ms=latency_ms, mode="VLLM")


def build_backend(settings: Settings) -> InferenceBackend:
    mode = settings.inference_mode.upper()
    if mode == "VLLM":
        return VLLMBackend(settings)
    return MockBackend(settings)
