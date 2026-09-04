"""Algorithm 1 — SKILL.state runtime (arXiv:2608.26263 §3.2).

At every step t the model receives ONLY A_t = (P, Σ_t, O_t).
It emits throwaway reasoning R_t plus a JSON object
``{"state_patch": ΔΣ_t, "action": a_t}``.

The runtime:
  1. extracts the JSON
  2. validates ΔΣ_t against the skill schema on a COPY of Σ (live Σ unchanged)
  3. validates action grammar
  4. executes a_t against the environment
  5. ONLY if the env accepted the action: commit Σ ← Σ ⊕ ΔΣ
  6. discards R_t permanently — it never re-enters the next prompt
  7. records prompt/completion tokens so we can prove O(1) prompt growth

Schema/JSON failure: do not merge, do not execute, retry once with the
validator error as the next observation. Two consecutive schema failures
end the episode as failed.

Env rejection (grammar-valid action, valid=false): leave Σ unchanged, pass
the env error as the next observation. That is not a schema failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from skillstate.json_extract import JsonExtractError, extract_json_object
from skillstate.logging_util import (
    EpisodeResult,
    StepResult,
    TokenTotals,
    diff_paths,
)
from skillstate.classify import classify_patch, item_shelf, summarize_episode
from skillstate.ollama_client import LLMClient
from skillstate.prompts import build_skill_state_prompt
from skillstate.schemas import PatchValidationError, apply_validated_patch, validate_model_output

OnToken = Callable[[str], None]
OnEvent = Callable[[dict[str, Any]], None]
PauseCheck = Callable[[], None]


def run_skill_state(
    *,
    skill_name: str,
    instructions: str,
    state_schema: type[BaseModel],
    initial_state: dict[str, Any],
    env: Any,
    llm: LLMClient,
    parse_action: Callable[[str], Any | None],
    max_steps: int,
    seed: int = 0,
    model: str = "",
    on_token: OnToken | None = None,
    on_event: OnEvent | None = None,
    pause_check: PauseCheck | None = None,
) -> EpisodeResult:
    state = dict(initial_state)
    observation = env.reset()
    totals = TokenTotals()
    steps: list[StepResult] = []
    consecutive_failures = 0
    failed = False
    fail_reason: str | None = None
    extra: dict[str, Any] = {}
    episode_t0 = time.perf_counter()

    def emit(event: dict[str, Any]) -> None:
        if on_event:
            on_event(event)

    for t in range(max_steps):
        if pause_check:
            pause_check()

        step_t0 = time.perf_counter()
        prompt = build_skill_state_prompt(instructions, state, observation)
        emit({"type": "prompt", "step": t, "prompt": prompt, "observation": observation, "state": state})

        token_buf: list[str] = []

        def _on_token(piece: str, buf: list[str] = token_buf) -> None:
            buf.append(piece)
            emit({"type": "token", "text": piece})
            if on_token:
                on_token(piece)

        llm_result = llm.complete(prompt, on_token=_on_token)
        totals.add(llm_result.prompt_tokens, llm_result.completion_tokens)

        state_before = dict(state)
        patch: dict[str, Any] = {}
        action = ""
        reasoning = llm_result.text
        next_observation = observation
        validation_error: str | None = None
        env_error = False
        done = False
        applied = False
        proposed_state: dict[str, Any] | None = None

        try:
            obj, reasoning = extract_json_object(llm_result.text)
            output = validate_model_output(obj)
            if parse_action(output.action) is None:
                raise PatchValidationError(
                    f"action does not match skill grammar: {output.action!r}"
                )
            # Schema-ok on a copy. Live Σ stays put until the env accepts a_t.
            proposed_state = apply_validated_patch(
                state, output.state_patch, state_schema
            )
            patch = output.state_patch
            action = output.action
            consecutive_failures = 0
            next_observation, done, info = env.step(action)
            env_error = not bool(info.get("valid", True))
            if not env_error:
                state = proposed_state
                applied = True
        except (JsonExtractError, PatchValidationError) as exc:
            validation_error = str(exc)
            consecutive_failures += 1
            next_observation = (
                "VALIDATOR ERROR: "
                f"{exc}. Re-emit a JSON object with exactly keys "
                "state_patch and action. The previous patch was NOT applied "
                "and no action was executed. Nested fields (e.g. shelves) "
                "belong under their parent key, not at the top level. "
                "The original observation still stands:\n"
                f"{observation}"
            )
            if consecutive_failures >= 2:
                failed = True
                fail_reason = (
                    "two consecutive validator failures: " + validation_error
                )
                done = True

        changed = diff_paths(state_before, state) if applied else []
        success_now = bool(env.success()) if not failed else False
        proposed_item_shelf, patch_class = _classify_step(
            env, proposed_state, validation_error, action
        )
        step = StepResult(
            step=t,
            prompt=prompt,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            observation=observation,
            reasoning=reasoning,
            state_patch=patch,
            action=action,
            state_before=state_before,
            state_after=dict(state),
            changed_keys=changed,
            next_observation=next_observation,
            validation_error=validation_error,
            env_error=env_error,
            done=done,
            success=success_now,
            totals=totals.as_dict(),
            proposed_item_shelf=proposed_item_shelf,
            patch_class=patch_class,
            wall_s=time.perf_counter() - step_t0,
            runtime="skillstate",
        )
        steps.append(step)
        emit({"type": "step", "step": step.as_dict()})

        _update_drift_recovery(env, state, t, extra)
        if done:
            break
        # Reasoning is discarded here: the next prompt is built from (P, Σ, O)
        # only. `reasoning` is kept on the StepResult for the log/UI.
        observation = next_observation

    success = (not failed) and bool(env.success())
    extra["wall_s"] = time.perf_counter() - episode_t0
    extra["env"] = env.snapshot() if hasattr(env, "snapshot") else {}
    rec = getattr(env, "silent_drift_record", None)
    if rec:
        extra.update(
            {
                "drift_step": rec.get("drift_step"),
                "drifted_item": rec.get("drifted_item"),
                "from_shelf": rec.get("from_shelf"),
                "to_shelf": rec.get("to_shelf"),
                "recovery_lag": extra.get("recovery_lag"),
            }
        )
    result = EpisodeResult(
        runtime="skillstate",
        skill=skill_name,
        seed=seed,
        model=model,
        success=success,
        failed=failed,
        fail_reason=fail_reason,
        steps=steps,
        totals=totals,
        extra=extra,
    )
    result.extra["classifier"] = summarize_episode(result)
    emit({"type": "done", "result": {k: v for k, v in result.as_dict().items() if k != "steps"}})
    return result


def _classify_step(
    env: Any,
    proposed_state: dict[str, Any] | None,
    validation_error: str | None,
    action: str,
) -> tuple[str | None, str | None]:
    rec = getattr(env, "silent_drift_record", None)
    if not rec or not rec.get("drifted_item"):
        if validation_error:
            return None, "grammar_fail"
        return None, None
    item = rec["drifted_item"]
    if validation_error or proposed_state is None:
        return None, "grammar_fail"
    shelf = item_shelf(proposed_state.get("inventory"), item)
    label = classify_patch(
        proposed_inventory=proposed_state.get("inventory"),
        item=item,
        from_shelf=rec.get("from_shelf") or "",
        to_shelf=rec.get("to_shelf") or "",
        action=action,
    )
    return shelf, label


def _item_shelf(inventory: Any, item: str) -> str | None:
    if not isinstance(inventory, dict):
        return None
    for shelf, value in inventory.items():
        if value == item:
            return shelf
    return None


def _update_drift_recovery(
    env: Any, state: dict[str, Any], step_index: int, extra: dict[str, Any]
) -> None:
    rec = getattr(env, "silent_drift_record", None)
    if not rec:
        return
    extra.setdefault("drift_step", rec.get("drift_step"))
    extra.setdefault("drifted_item", rec.get("drifted_item"))
    extra.setdefault("from_shelf", rec.get("from_shelf"))
    extra.setdefault("to_shelf", rec.get("to_shelf"))
    if extra.get("recovery_lag") is not None:
        return
    drift_step = rec.get("drift_step")
    item = rec.get("drifted_item")
    if item is None or drift_step is None or step_index <= drift_step:
        extra.setdefault("recovery_lag", None)
        return
    gt = env.snapshot() if hasattr(env, "snapshot") else {}
    sigma_shelf = _item_shelf(state.get("inventory"), item)
    gt_shelf = _item_shelf(gt.get("inventory"), item)
    if sigma_shelf == gt_shelf:
        extra["recovery_lag"] = step_index - drift_step
    else:
        extra.setdefault("recovery_lag", None)
