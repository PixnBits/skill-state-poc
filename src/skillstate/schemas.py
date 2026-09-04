"""Pydantic v2 schemas shared by the runtime.

The model output contract is non-negotiable (paper Appendix A.4):

    { "state_patch": { ... }, "action": "<cmd>" }

Exactly those two keys. Extra keys fail validation; the patch is not applied
and the action is not executed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from skillstate.merge import MergeError, deep_merge


class PatchValidationError(ValueError):
    """JSON / schema / action-grammar failure. Runtime must not apply or execute."""


class ModelOutput(BaseModel):
    """Validated LLM payload. ``extra='forbid'`` is load-bearing."""

    model_config = ConfigDict(extra="forbid")

    state_patch: dict[str, Any]
    action: str

    @field_validator("action")
    @classmethod
    def action_is_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("action must be a non-empty string")
        return stripped


def validate_model_output(obj: Any) -> ModelOutput:
    if not isinstance(obj, dict):
        raise PatchValidationError(
            f"model output is not a JSON object (got {type(obj).__name__})"
        )
    keys = set(obj.keys())
    if keys != {"state_patch", "action"}:
        raise PatchValidationError(
            "JSON must have exactly two keys: state_patch and action; "
            f"got {sorted(keys)}"
        )
    if not isinstance(obj["state_patch"], dict):
        raise PatchValidationError(
            "state_patch must be a JSON object, "
            f"got {type(obj['state_patch']).__name__}"
        )
    if not isinstance(obj["action"], str):
        raise PatchValidationError(
            f"action must be a string, got {type(obj['action']).__name__}"
        )
    try:
        return ModelOutput.model_validate(obj)
    except ValidationError as exc:
        raise PatchValidationError(f"model output failed schema: {exc}") from exc


def schema_top_keys(schema: type[BaseModel]) -> set[str]:
    return set(schema.model_fields.keys())


def apply_validated_patch(
    state: dict[str, Any],
    patch: dict[str, Any],
    schema: type[BaseModel],
) -> dict[str, Any]:
    """Σ ⊕ ΔΣ then pydantic-validate. Never mutates ``state``.

    Unknown top-level keys are refused (merge allow-list **and** extra='forbid').
    Nested validation (shelf ids, types) is the skill schema’s job.
    """
    allowed = schema_top_keys(schema)
    try:
        merged = deep_merge(state, patch, allowed_top_keys=allowed)
    except MergeError as exc:
        raise PatchValidationError(str(exc)) from exc
    try:
        model = schema.model_validate(merged)
    except ValidationError as exc:
        raise PatchValidationError(
            f"merged state failed schema validation: {exc}"
        ) from exc
    return model.model_dump(mode="json")
