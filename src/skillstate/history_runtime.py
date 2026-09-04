"""Honest ReAct baseline: append every observation, thought, and action.

Prompt length is O(t) by construction. If this curve is not strictly growing
while SKILL.state stays flat, the implementation is wrong.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from skillstate.json_extract import JsonExtractError, extract_react_action
from skillstate.logging_util import EpisodeResult, StepResult, TokenTotals, diff_paths
from skillstate.ollama_client import LLMClient
from skillstate.prompts import build_react_prompt, format_history_turn

OnToken = Callable[[str], None]
OnEvent = Callable[[dict[str, Any]], None]
PauseCheck = Callable[[], None]


def run_history(
    *,
    skill_name: str,
    instructions: str,
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
    # History runtime does not maintain Σ as a sufficient statistic. We keep a
    # dummy state dict only so the compare table / UI have a comparable snapshot
    # of the *environment* — it is NOT inserted into the prompt.
    observation = env.reset()
    history: list[str] = []
    totals = TokenTotals()
    steps: list[StepResult] = []
    consecutive_failures = 0
    failed = False
    fail_reason: str | None = None
    extra: dict[str, Any] = {
        "history_turns": 0,
        "recovery_via": "env_gt_via_actions",
    }
    episode_t0 = time.perf_counter()

    def emit(event: dict[str, Any]) -> None:
        if on_event:
            on_event(event)

    for t in range(max_steps):
        if pause_check:
            pause_check()

        step_t0 = time.perf_counter()
        prompt = build_react_prompt(instructions, history, observation)
        emit({"type": "prompt", "step": t, "prompt": prompt, "observation": observation})

        def _on_token(piece: str) -> None:
            emit({"type": "token", "text": piece})
            if on_token:
                on_token(piece)

        llm_result = llm.complete(prompt, on_token=_on_token)
        totals.add(llm_result.prompt_tokens, llm_result.completion_tokens)

        action = ""
        reasoning = llm_result.text
        next_observation = observation
        validation_error: str | None = None
        env_error = False
        done = False
        gt_before = env.snapshot() if hasattr(env, "snapshot") else {}

        try:
            action, reasoning = extract_react_action(llm_result.text)
            if parse_action(action) is None:
                raise JsonExtractError(f"action does not match skill grammar: {action!r}")
            consecutive_failures = 0
            next_observation, done, info = env.step(action)
            env_error = not bool(info.get("valid", True))
        except JsonExtractError as exc:
            validation_error = str(exc)
            consecutive_failures += 1
            action = ""
            next_observation = (
                "VALIDATOR ERROR: "
                f"{exc}. Reply with reasoning and a line 'Action: <cmd>'."
            )
            if consecutive_failures >= 2:
                failed = True
                fail_reason = "two consecutive validator failures: " + validation_error
                done = True

        # Append this turn to history even on validator failure (honest ReAct:
        # the transcript grows with every model call).
        history.append(format_history_turn(observation, llm_result.text))

        gt_after = env.snapshot() if hasattr(env, "snapshot") else {}
        step = StepResult(
            step=t,
            prompt=prompt,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            observation=observation,
            reasoning=reasoning,
            state_patch={},
            action=action,
            state_before=gt_before,
            state_after=gt_after,
            changed_keys=diff_paths(gt_before, gt_after),
            next_observation=next_observation,
            validation_error=validation_error,
            env_error=env_error,
            done=done,
            success=bool(env.success()) if not failed else False,
            totals=totals.as_dict(),
            wall_s=time.perf_counter() - step_t0,
            runtime="history",
        )
        steps.append(step)
        emit({"type": "step", "step": step.as_dict()})
        _update_history_drift_recovery(env, action, env_error, t, observation, extra)
        if done:
            break
        observation = next_observation

    extra["history_turns"] = len(history)
    extra["wall_s"] = time.perf_counter() - episode_t0
    extra["env"] = env.snapshot() if hasattr(env, "snapshot") else {}
    rec = getattr(env, "silent_drift_record", None)
    if rec:
        extra.setdefault("drift_step", rec.get("drift_step"))
        extra.setdefault("drifted_item", rec.get("drifted_item"))
        extra.setdefault("from_shelf", rec.get("from_shelf"))
        extra.setdefault("to_shelf", rec.get("to_shelf"))
        extra.setdefault("recovery_lag", extra.get("recovery_lag"))
    result = EpisodeResult(
        runtime="history",
        skill=skill_name,
        seed=seed,
        model=model,
        success=(not failed) and bool(env.success()),
        failed=failed,
        fail_reason=fail_reason,
        steps=steps,
        totals=totals,
        extra=extra,
    )
    emit({"type": "done", "result": {k: v for k, v in result.as_dict().items() if k != "steps"}})
    return result


def _action_targets_to_shelf(action: str, item: str, to_shelf: str) -> bool:
    parts = (action or "").split()
    if len(parts) == 3 and parts[0].upper() == "SHIP" and parts[1] == item and parts[2] == to_shelf:
        return True
    if (
        len(parts) == 4
        and parts[0].upper() == "MOVE"
        and parts[1] == item
        and parts[3] == to_shelf
    ):
        return True
    return False


def _update_history_drift_recovery(
    env: Any,
    action: str,
    env_error: bool,
    step_index: int,
    observation: str,
    extra: dict[str, Any],
) -> None:
    """Recovery from env ground truth via actions — history has no Σ.

    After silent drift the item already sits on to_shelf in the env. Recovery
    is the first later step where a *valid* action names that shelf for the
    item, or the item has been shipped. We do not invent a patch_class.
    """
    rec = getattr(env, "silent_drift_record", None)
    if not rec or not rec.get("drifted_item"):
        return
    extra.setdefault("drift_step", rec.get("drift_step"))
    extra.setdefault("drifted_item", rec.get("drifted_item"))
    extra.setdefault("from_shelf", rec.get("from_shelf"))
    extra.setdefault("to_shelf", rec.get("to_shelf"))
    extra.setdefault("recovery_lag", None)
    extra.setdefault("first_post_scanner_targeted_to_shelf", None)
    extra["recovery_via"] = "env_gt_via_actions"
    item = rec["drifted_item"]
    to_shelf = rec.get("to_shelf") or ""
    drift_step = rec.get("drift_step")
    if drift_step is None or step_index <= int(drift_step):
        return
    saw_scanner = "Cycle count:" in (observation or "") or "Floor scanner:" in (
        observation or ""
    )
    targeted = _action_targets_to_shelf(action, item, to_shelf)
    if saw_scanner and extra.get("first_post_scanner_targeted_to_shelf") is None:
        extra["first_post_scanner_targeted_to_shelf"] = targeted
    if extra.get("recovery_lag") is not None:
        return
    snap = env.snapshot() if hasattr(env, "snapshot") else {}
    shipped = item in (snap.get("shipped") or [])
    valid_target = (not env_error) and targeted
    # loc == to_shelf is true immediately after drift without an action — ignore.
    if shipped or valid_target:
        extra["recovery_lag"] = step_index - int(drift_step)
