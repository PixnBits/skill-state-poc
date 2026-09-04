"""Deterministic stand-in for Ollama. Used by offline tests and --offline demos."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from skillstate.logging_util import estimate_tokens
from skillstate.ollama_client import LLMResult


class FakeLLM:
    """Wrap a ``prompt -> completion`` callable."""

    def __init__(self, responder: Callable[[str], str], model: str = "fake") -> None:
        self.responder = responder
        self.model = model
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResult:
        self.prompts.append(prompt)
        text = self.responder(prompt)
        if on_token:
            on_token(text)
        return LLMResult(
            text=text,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
        )


class SequenceLLM:
    """Return canned completions in order. Useful for validator-retry tests."""

    def __init__(self, responses: Sequence[str], model: str = "fake-seq") -> None:
        self.responses = list(responses)
        self.model = model
        self.prompts: list[str] = []
        self._i = 0

    def complete(
        self,
        prompt: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResult:
        self.prompts.append(prompt)
        if self._i >= len(self.responses):
            raise RuntimeError("SequenceLLM exhausted canned responses")
        text = self.responses[self._i]
        self._i += 1
        if on_token:
            on_token(text)
        return LLMResult(
            text=text,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
        )
