"""Token accounting and episode traces. No telemetry leaves the machine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def estimate_tokens(text: str) -> int:
    """Fallback when the API does not return ``usage``.

    ~4 characters per token is the usual English heuristic. Good enough to
    prove O(1) vs O(T) prompt growth; Ollama's ``usage`` is preferred live.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def diff_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    """Return dotted paths whose values changed between two JSON-like trees."""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(diff_paths(before[key], after[key], path))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        return [prefix or "$"]
    return [prefix or "$"]


@dataclass
class TokenTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_prompt_tokens: int = 0
    steps: int = 0
    prompt_curve: list[int] = field(default_factory=list)

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_tokens)
        self.steps += 1
        self.prompt_curve.append(prompt_tokens)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_prompt_tokens(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.prompt_tokens / self.steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "max_prompt_tokens": self.max_prompt_tokens,
            "steps": self.steps,
            "prompt_curve": list(self.prompt_curve),
        }


@dataclass
class StepResult:
    step: int
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    observation: str
    reasoning: str
    state_patch: dict[str, Any]
    action: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    changed_keys: list[str]
    next_observation: str
    validation_error: str | None
    env_error: bool
    done: bool
    success: bool
    totals: dict[str, Any]
    proposed_item_shelf: str | None = None
    patch_class: str | None = None
    runtime: str = "skillstate"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class EpisodeResult:
    runtime: str
    skill: str
    seed: int
    model: str
    success: bool
    failed: bool
    fail_reason: str | None
    steps: list[StepResult] = field(default_factory=list)
    totals: TokenTotals = field(default_factory=TokenTotals)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "skill": self.skill,
            "seed": self.seed,
            "model": self.model,
            "success": self.success,
            "failed": self.failed,
            "fail_reason": self.fail_reason,
            "totals": self.totals.as_dict(),
            "steps": [s.as_dict() for s in self.steps],
            "extra": self.extra,
        }


def compact_state(state: dict[str, Any]) -> str:
    """Human row for the CLI: drop empty shelves so the table fits a terminal."""
    payload = dict(state)
    inventory = payload.get("inventory")
    if isinstance(inventory, dict):
        occupied = {k: v for k, v in inventory.items() if v is not None}
        payload["inventory"] = occupied
        payload["empty_shelves"] = sum(1 for v in inventory.values() if v is None)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_run(
    payload: dict[str, Any],
    directory: Path | None = None,
    suffix: str = "",
) -> Path:
    runs_dir = directory or Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = payload.get("runtime") or payload.get("cli") or suffix or "run"
    path = runs_dir / f"{stamp}-{tag}.json"
    n = 1
    while path.exists():
        n += 1
        path = runs_dir / f"{stamp}-{tag}-{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def summarize_curve(curve: Iterable[int]) -> str:
    values = list(curve)
    if not values:
        return "(empty)"
    return " → ".join(str(v) for v in values)
