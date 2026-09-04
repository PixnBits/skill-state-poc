"""Ollama via the OpenAI-compatible HTTP API. Local only, no tools, T=0."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from skillstate.logging_util import estimate_tokens

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:14b"

INSTALL_HELP = """
Ollama is not reachable at {base}

Install:
  Linux / Framework Desktop:
    curl -fsSL https://ollama.com/install.sh | sh
  Windows:
    winget install Ollama.Ollama
    (or download https://ollama.com/download/windows)

Start the daemon:
  ollama serve

Pull a model (Framework 32/64GB default):
  ollama pull {model}

Suggested models by unified memory:
  32GB  qwen2.5:14b   or llama3.1:8b
  64GB  qwen2.5:32b   or qwen3:30b
  128GB llama3.3:70b  or qwen2.5:72b

Then re-run this command. This PoC talks only to 127.0.0.1 — no cloud APIs.
""".strip()


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    raw_usage: dict[str, Any] | None = None


class LLMClient:
    def complete(
        self,
        prompt: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResult:
        raise NotImplementedError


def default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def default_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _root_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


def health_check(
    base_url: str | None = None,
    model: str | None = None,
    *,
    require_model: bool = True,
) -> dict[str, Any]:
    """Return tags payload or raise SystemExit(1) with install instructions."""
    base = (base_url or default_base_url()).rstrip("/")
    model_name = model or default_model()
    tags_url = _root_url(base) + "/api/tags"
    try:
        response = httpx.get(tags_url, timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(INSTALL_HELP.format(base=_root_url(base) or "http://127.0.0.1:11434", model=model_name))
        print(f"\nUnderlying error: {exc}")
        raise SystemExit(1) from exc
    payload = response.json()
    names = [m.get("name", "") for m in payload.get("models", [])]
    if require_model and not _model_present(model_name, names):
        root = _root_url(base)
        print(
            f"Ollama is running at {root} but model {model_name!r} is not pulled.\n"
            f"Installed: {names or '(none)'}\n\n"
            f"  ollama pull {model_name}\n"
        )
        raise SystemExit(1)
    return {"ok": True, "models": names, "model": model_name, "base_url": base}


def _model_present(model: str, names: list[str]) -> bool:
    if model in names:
        return True
    # qwen2.5:14b matches qwen2.5:14b-instruct-q4_K_M etc.
    return any(n == model or n.startswith(model + "-") or n.startswith(model + ":") for n in names)


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model or default_model()
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        prompt: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": True,
        }
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                    else:
                        data = line.strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content") or ""
                    if not token:
                        message = choices[0].get("message") or {}
                        token = message.get("content") or ""
                    if token:
                        pieces.append(token)
                        if on_token:
                            on_token(token)
        except httpx.HTTPError as exc:
            print(INSTALL_HELP.format(base=_root_url(self.base_url), model=self.model))
            print(f"\nRequest error: {exc}")
            raise SystemExit(1) from exc

        text = "".join(pieces)
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(prompt))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_usage=usage or None,
        )
