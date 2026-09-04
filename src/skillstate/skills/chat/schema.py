"""Chat execution-state schema. extra='forbid' is load-bearing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatState(BaseModel):
    """Σ for the split-view chat skill. Sufficient statistic for the conversation."""

    model_config = ConfigDict(extra="forbid")

    goal: str = ""
    facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tone: str = "direct"
    last_action: str = ""

    @field_validator("facts", "open_questions", "decisions", mode="before")
    @classmethod
    def null_list_means_empty(cls, value: Any) -> Any:
        # RFC 7396 null deletes the key; missing keys then take the default [].
        # If a patch explicitly sets the field to null at the typed layer, treat
        # that as "clear the list" rather than a schema failure.
        if value is None:
            return []
        return value

    @field_validator("facts", "open_questions", "decisions")
    @classmethod
    def short_strings(cls, value: list[Any]) -> list[str]:
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("list entries must be strings")
            text = item.strip()
            if text:
                out.append(text)
        return out

    @field_validator("tone")
    @classmethod
    def tone_nonempty(cls, value: str) -> str:
        stripped = (value or "").strip()
        return stripped or "direct"


def initial_chat_state() -> dict[str, Any]:
    return ChatState().model_dump(mode="json")
