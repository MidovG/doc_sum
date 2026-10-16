"""Тонкий слой над LLM-провайдерами + дисковый кэш ответов.

Провайдеры:
  openai     — OpenAI и любой OpenAI-совместимый API (vLLM, LM Studio, OpenRouter,
               DeepSeek, YandexGPT/GigaChat через совместимые шлюзы) — задаётся --base-url
  ollama     — локальные модели через Ollama (родной API, с явным размером контекста)
  anthropic  — Claude
  fake       — без сети, для тестов и проверки пайплайна
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


class LLM(ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2000,
                 cache_dir: Path | None = None, retries: int = 4):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _complete(self, system: str, user: str) -> str: ...

    def complete(self, system: str, user: str) -> str:
        key = hashlib.sha256(
            json.dumps([self.name, self.model, self.temperature, system, user]).encode()
        ).hexdigest()
        cached = self.cache_dir / f"{key}.txt" if self.cache_dir else None
        if cached and cached.exists():
            return cached.read_text(encoding="utf-8")

        delay = 2.0
        for attempt in range(1, self.retries + 1):
            try:
                answer = self._complete(system, user).strip()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == self.retries or not _is_retryable(e):
                    raise
                log.warning("LLM ошибка (%s), повтор %d/%d через %.0fс",
                            e, attempt, self.retries, delay)
                time.sleep(delay)
                delay *= 2
        if cached:
            cached.write_text(answer, encoding="utf-8")
        return answer


def _is_retryable(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return True  # сетевые ошибки/таймауты


class OpenAICompatible(LLM):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kw):
        super().__init__(model, **kw)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            timeout=600,
        )

    def _complete(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = r.choices[0].message.content or ""
        # Reasoning-модели (qwen3, deepseek-r1) могут вернуть рассуждения в <think>
        return re.sub(r"(?s)<think>.*?</think>", "", text)


class Ollama(LLM):
    """Локальные модели через родной API Ollama (/api/chat).

    Родной API, а не OpenAI-совместимый /v1, нужен ради options.num_ctx: по умолчанию
    Ollama держит маленький контекст и молча обрезает длинный промпт — модель видит
    только конец документа. Здесь контекст задаётся явно на каждый запрос.
    """

    name = "ollama"

    def __init__(self, model: str, base_url: str | None = None, num_ctx: int = 16_384,
                 timeout: float = 1800, **kw):
        super().__init__(model, **kw)
        import httpx

        self.base_url = _ollama_base(base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434")
        self.num_ctx = num_ctx
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)

    def check(self) -> None:
        """Быстрая проверка до начала работы: сервер запущен, модель скачана."""
        import httpx

        try:
            tags = self.http.get("/api/tags", timeout=10).json().get("models", [])
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Ollama недоступна по адресу {self.base_url} ({e}). "
                "Запустите её: `ollama serve` или приложение Ollama."
            ) from None
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            have = ", ".join(sorted(n for n in names if n)) or "нет ни одной"
            raise RuntimeError(
                f"Модель {self.model!r} не найдена в Ollama. Скачайте: `ollama pull {self.model}`. "
                f"Установлены: {have}"
            )

    def _complete(self, system: str, user: str) -> str:
        r = self.http.post("/api/chat", json={
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        })
        r.raise_for_status()
        data = r.json()
        prompt_tokens = data.get("prompt_eval_count")
        if prompt_tokens and prompt_tokens >= self.num_ctx - 16:
            log.warning("Промпт (%d токенов) упёрся в num_ctx=%d — часть текста могла обрезаться. "
                        "Увеличьте --num-ctx или уменьшите --chunk-chars/--context-chars.",
                        prompt_tokens, self.num_ctx)
        text = data.get("message", {}).get("content", "")
        return re.sub(r"(?s)<think>.*?</think>", "", text)  # qwen3/deepseek-r1 рассуждения


def _ollama_base(url: str) -> str:
    url = url.strip().rstrip("/")
    if "://" not in url:  # OLLAMA_HOST бывает вида "0.0.0.0:11434"
        url = "http://" + url
    url = url.replace("://0.0.0.0", "://127.0.0.1")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


class Anthropic(LLM):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, **kw):
        super().__init__(model, **kw)
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class Fake(LLM):
    """Детерминированная «модель» без сети: возвращает начало входного текста."""

    name = "fake"

    def _complete(self, system: str, user: str) -> str:
        body = user.split("<<<", 1)[-1].rsplit(">>>", 1)[0]
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
        return "- " + "\n- ".join(s for s in sentences[:4] if s) if body.strip() else "- (пусто)"


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-sonnet-4-5",
    "fake": "fake",
}

PROVIDERS = {"openai": OpenAICompatible, "ollama": Ollama, "anthropic": Anthropic, "fake": Fake}


def make_llm(provider: str, model: str | None = None, base_url: str | None = None,
             api_key: str | None = None, num_ctx: int | None = None, **kw) -> LLM:
    if provider not in PROVIDERS:
        raise ValueError(f"Неизвестный провайдер {provider!r}; доступны: {', '.join(PROVIDERS)}")
    cls = PROVIDERS[provider]
    model = model or DEFAULT_MODELS[provider]
    if cls is Fake:
        return Fake(model, **kw)
    if cls is Anthropic:
        return Anthropic(model, api_key=api_key, **kw)
    if cls is Ollama:
        return Ollama(model, base_url=base_url, **({"num_ctx": num_ctx} if num_ctx else {}), **kw)
    return OpenAICompatible(model, base_url=base_url, api_key=api_key, **kw)
