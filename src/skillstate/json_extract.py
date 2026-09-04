"""Messy-model-tolerant extraction of the SKILL.state JSON payload.

Local 8B–14B models (qwen, llama) routinely wrap the object in prose, fences,
or trailing commentary. We try, in order:

1. A fenced ``json`` (or bare) code block whose body parses as an object.
2. The first balanced ``{...}`` that contains both required keys.
3. The first balanced object that parses at all.

Trailing commas are stripped before ``json.loads``. The text *outside* the
chosen object is returned as throwaway reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any


class JsonExtractError(ValueError):
    """No usable JSON object in the completion."""


_FENCE_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SINGLE_QUOTED_KEYS = re.compile(r"(?<!\w)'([A-Za-z_][A-Za-z0-9_]*)'\s*:")
_NONE_LITERAL = re.compile(r"\bNone\b")
_TRUE_LITERAL = re.compile(r"\bTrue\b")
_FALSE_LITERAL = re.compile(r"\bFalse\b")


def extract_json_object(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(object, reasoning)``. ``reasoning`` is everything but the JSON."""
    if not text or not text.strip():
        raise JsonExtractError("empty model output")

    candidates: list[tuple[int, int, str]] = []

    for match in _FENCE_JSON.finditer(text):
        body = match.group(1).strip()
        if body.startswith("{"):
            candidates.append((match.start(), match.end(), body))

    for start, end, body in _iter_balanced_objects(text):
        candidates.append((start, end, body))

    seen: set[str] = set()
    parsed_with_keys: list[tuple[dict[str, Any], str]] = []
    parsed_any: list[tuple[dict[str, Any], str]] = []

    for start, end, body in candidates:
        if body in seen:
            continue
        seen.add(body)
        obj = _try_parse(body)
        if not isinstance(obj, dict):
            continue
        reasoning = _reasoning_around(text, start, end)
        if "state_patch" in obj and "action" in obj:
            parsed_with_keys.append((obj, reasoning))
        else:
            parsed_any.append((obj, reasoning))

    if parsed_with_keys:
        return parsed_with_keys[0]
    if parsed_any:
        return parsed_any[0]
    raise JsonExtractError("no JSON object found in model output")


def extract_react_action(text: str) -> tuple[str, str]:
    """Parse ReAct ``Action: <cmd>``. Returns ``(action, reasoning)``."""
    if not text:
        raise JsonExtractError("empty model output")
    # Prefer the last Action: line (models sometimes draft then correct).
    matches = list(re.finditer(r"(?im)^\s*Action\s*:\s*(.+?)\s*$", text))
    if matches:
        action = matches[-1].group(1).strip().strip("`").strip('"').strip("'")
        reasoning = (text[: matches[-1].start()] + text[matches[-1].end() :]).strip()
        if action:
            return action, reasoning
    # Fallback: SKILL.state JSON if the model ignored ReAct format.
    try:
        obj, reasoning = extract_json_object(text)
    except JsonExtractError as exc:
        raise JsonExtractError(
            "no 'Action: <cmd>' line and no JSON object with an action key"
        ) from exc
    action = obj.get("action")
    if not isinstance(action, str) or not action.strip():
        raise JsonExtractError("JSON fallback has no string action")
    return action.strip(), reasoning


def _try_parse(body: str) -> dict[str, Any] | None:
    for candidate in _repair_variants(body):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _repair_variants(body: str) -> list[str]:
    stripped = body.strip()
    variants = [stripped]
    no_commas = _TRAILING_COMMA.sub(r"\1", stripped)
    if no_commas != stripped:
        variants.append(no_commas)
    pythonic = _FALSE_LITERAL.sub(
        "false",
        _TRUE_LITERAL.sub("true", _NONE_LITERAL.sub("null", no_commas)),
    )
    pythonic = _SINGLE_QUOTED_KEYS.sub(r'"\1":', pythonic)
    if pythonic not in variants:
        variants.append(pythonic)
    return variants


def _iter_balanced_objects(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        end = _match_braces(text, i)
        if end is not None:
            found.append((i, end, text[i:end]))
            i = end
        else:
            i += 1
    return found


def _match_braces(text: str, start: int) -> int | None:
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _reasoning_around(text: str, start: int, end: int) -> str:
    return (text[:start] + text[end:]).strip()
