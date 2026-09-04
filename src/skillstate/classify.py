"""Classify a proposed Σ against a silent-drift ground-truth move.

Works on the *proposed* inventory (schema-valid merge of Σ ⊕ ΔΣ), even when
the env later rejects the action and the commit rule discards the patch.
"""

from __future__ import annotations

from typing import Any

from skillstate.logging_util import EpisodeResult, StepResult

GRAMMAR_FAIL = "grammar_fail"
CORRECT_RELOCATION = "correct_relocation"
DELETE_ONLY = "delete_only"
STALE_LOCATION = "stale_location"
STARVED_CORRECT = "starved_correct"


def item_shelf(inventory: dict[str, Any] | None, item: str) -> str | None:
    if not inventory or not item:
        return None
    for shelf, value in inventory.items():
        if value == item:
            return shelf
    return None


def _ship_target(action: str, item: str) -> str | None:
    parts = (action or "").split()
    if len(parts) == 3 and parts[0].upper() == "SHIP" and parts[1] == item:
        return parts[2]
    return None


def classify_patch(
    *,
    proposed_inventory: dict[str, Any] | None,
    item: str,
    from_shelf: str,
    to_shelf: str,
    grammar_fail: bool = False,
    action: str = "",
) -> str:
    """Label where the drifted item sits in the proposed inventory.

    ``SHIP <item> <to_shelf>`` with the item absent from inventory is a
    relocate-and-ship (the model named the scanner shelf), not delete_only.
    ``SHIP <item> <from_shelf>`` plus nulling from_shelf is delete_only.
    """
    if grammar_fail or proposed_inventory is None:
        return GRAMMAR_FAIL
    loc = item_shelf(proposed_inventory, item)
    ship_shelf = _ship_target(action, item)
    if loc == to_shelf or ship_shelf == to_shelf:
        return CORRECT_RELOCATION
    if loc is None:
        return DELETE_ONLY
    return STALE_LOCATION


def is_starved_correct(patch_class: str, env_error: bool) -> bool:
    return patch_class == CORRECT_RELOCATION and bool(env_error)


def first_post_scanner_step(steps: list[StepResult] | list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in steps:
        d = step.as_dict() if isinstance(step, StepResult) else step
        obs = d.get("observation") or ""
        if "Cycle count:" in obs or "Floor scanner:" in obs:
            return d
    return None


def summarize_episode(result: EpisodeResult) -> dict[str, Any]:
    extra = result.extra or {}
    drift_step = extra.get("drift_step")
    reached = drift_step is not None
    grammar_fail_count = 0
    delete_only = 0
    stale_location = 0
    correct_relocation = 0
    starved_correct = 0
    recovered_at = extra.get("recovery_lag")
    last_count_step = None
    if reached and recovered_at is not None:
        last_count_step = int(drift_step) + int(recovered_at)
    for step in result.steps:
        pc = step.patch_class
        if step.validation_error or pc == GRAMMAR_FAIL:
            grammar_fail_count += 1
        if not reached or step.step <= int(drift_step):
            continue
        if last_count_step is not None and step.step > last_count_step:
            continue
        if pc == DELETE_ONLY:
            delete_only += 1
        elif pc == STALE_LOCATION:
            stale_location += 1
        elif pc == CORRECT_RELOCATION:
            correct_relocation += 1
            if step.env_error:
                starved_correct += 1
    curve = list(result.totals.prompt_curve)
    prompt_flat = True
    if curve:
        prompt_flat = max(curve) <= min(curve) * 1.35 + 80
    scanner = first_post_scanner_step(result.steps)
    return {
        "model": result.model,
        "reached_drift": reached,
        "steps": result.totals.steps,
        "fail_reason": result.fail_reason,
        "recovery_lag": extra.get("recovery_lag"),
        "grammar_fail_count": grammar_fail_count,
        "delete_only": delete_only,
        "stale_location": stale_location,
        "correct_relocation": correct_relocation,
        "starved_correct": starved_correct,
        "success": result.success,
        "failed": result.failed,
        "prompt_flat": prompt_flat,
        "prompt_curve": curve,
        "drift_step": drift_step,
        "drifted_item": extra.get("drifted_item"),
        "from_shelf": extra.get("from_shelf"),
        "to_shelf": extra.get("to_shelf"),
        "first_post_scanner": None
        if scanner is None
        else {
            "step": scanner.get("step"),
            "action": scanner.get("action"),
            "state_patch": scanner.get("state_patch"),
            "patch_class": scanner.get("patch_class"),
            "proposed_item_shelf": scanner.get("proposed_item_shelf"),
            "env_error": scanner.get("env_error"),
            "observation": scanner.get("observation"),
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "reached_drift",
        "steps",
        "fail_reason",
        "recovery_lag",
        "grammar_fail_count",
        "delete_only",
        "stale_location",
        "correct_relocation",
        "starved_correct",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for key in headers:
            value = row.get(key)
            if value is None:
                cells.append("null")
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            else:
                text = str(value).replace("|", "\\|").replace("\n", " ")
                if len(text) > 80:
                    text = text[:77] + "..."
                cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
