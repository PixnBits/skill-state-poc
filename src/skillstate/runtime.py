"""Algorithm 1 — SKILL.state runtime (arXiv:2608.26263 §3.2).

At every step t the model receives ONLY A_t = (P, Σ_t, O_t).
It emits throwaway reasoning R_t plus a JSON object
``{"state_patch": ΔΣ_t, "action": a_t}``.

The runtime:
  1. extracts and validates the JSON against the skill schema
  2. applies Σ_{t+1} = Σ_t ⊕ ΔΣ_t  (deep merge; JSON null deletes)
  3. executes a_t against the environment
  4. discards R_t permanently — it never re-enters the next prompt
  5. records prompt/completion tokens so we can prove O(1) prompt growth

Validator failure: do not apply the patch, do not execute the action, retry
once with the validator error as the next observation. Two consecutive
failures end the episode as failed.
"""

from __future__ import annotations

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

    def emit(event: dict[str, Any]) -> None:
        if on_event:
            on_event(event)

    for t in range(max_steps):
        if pause_check:
            pause_check()

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

        try:
            obj, reasoning = extract_json_object(llm_result.text)
            output = validate_model_output(obj)
            if parse_action(output.action) is None:
                raise PatchValidationError(
                    f"action does not match skill grammar: {output.action!r}"
                )
            new_state = apply_validated_patch(state, output.state_patch, state_schema)
            patch = output.state_patch
            action = output.action
            state = new_state
            applied = True
            consecutive_failures = 0
            next_observation, done, info = env.step(action)
            env_error = not bool(info.get("valid", True))
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
            runtime="skillstate",
        )
        steps.append(step)
        emit({"type": "step", "step": step.as_dict()})

        if done:
            break
        # Reasoning is discarded here: the next prompt is built from (P, Σ, O)
        # only. `reasoning` is kept on the StepResult for the log/UI.
        observation = next_observation

    success = (not failed) and bool(env.success())
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
        extra={"env": env.snapshot() if hasattr(env, "snapshot") else {}},
    )
    emit({"type": "done", "result": {k: v for k, v in result.as_dict().items() if k != "steps"}})
    return result
