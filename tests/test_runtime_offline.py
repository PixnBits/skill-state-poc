"""Fake-LLM tests. No Ollama required."""

from __future__ import annotations

import json

from skillstate.fake_llm import FakeLLM, SequenceLLM
from skillstate.history_runtime import run_history
from skillstate.json_extract import extract_json_object
from skillstate.policies import warehouse_history_policy, warehouse_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills.warehouse import WarehouseSkill


def _run_ss(llm, *, compact=True, max_steps=8, seed=1):
    skill = WarehouseSkill()
    env = skill.make_env(seed, compact=compact, horizon=max_steps)
    return run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=env,
        llm=llm,
        parse_action=skill.parse_action,
        max_steps=max_steps,
        seed=seed,
        model="fake",
    )


def test_five_step_compact_episode_succeeds():
    result = _run_ss(FakeLLM(warehouse_skillstate_policy), compact=True, max_steps=8)
    assert result.success, result.fail_reason
    actions = [s.action for s in result.steps]
    assert actions[0].startswith("STORE")
    assert actions[1].startswith("STORE")
    assert actions[2].startswith("SHIP")
    assert actions[3].startswith("SHIP")
    assert actions[4] == "DONE"
    assert len(result.steps) == 5


def test_prompt_is_only_p_state_observation():
    result = _run_ss(FakeLLM(warehouse_skillstate_policy), compact=True, max_steps=8)
    for i, step in enumerate(result.steps):
        prompt = step.prompt
        assert "Skill Execution State:" in prompt
        assert "Latest Observation:" in prompt
        assert prompt.count("Latest Observation:") == 1
        assert "History:" not in prompt
        if i == 0:
            continue
        prev = result.steps[i - 1]
        # Throwaway reasoning must not re-enter A_{t+1}.
        if prev.reasoning.strip():
            assert prev.reasoning.strip() not in prompt
        # Prior observation text is not appended (it may share item ids with Σ).
        assert "Observation: " + prev.observation not in prompt
        assert "Reasoning & Action:" not in prompt


def test_json_extract_messy_wrappers():
    samples = [
        'Sure.\n```json\n{"state_patch": {"step": 1}, "action": "WAIT"}\n```\nhope that helps',
        '{"state_patch": {"step": 1}, "action": "WAIT"} trailing junk',
        "Reasoning here\n```\n{\"state_patch\": {\"step\": 1,}, \"action\": \"WAIT\"}\n```",
        "Here you go:\n{'state_patch': {'step': 1}, 'action': 'WAIT'}",
    ]
    # The single-quoted sample may or may not parse; fenced + trailing comma must.
    obj, reasoning = extract_json_object(samples[0])
    assert obj["action"] == "WAIT"
    assert "Sure" in reasoning
    obj, _ = extract_json_object(samples[1])
    assert obj["state_patch"]["step"] == 1
    obj, _ = extract_json_object(samples[2])
    assert obj["action"] == "WAIT"


def test_validator_retry_then_success():
    good = (
        'ok\n```json\n{"state_patch": {"last_action": "WAIT", "step": 1}, '
        '"action": "WAIT"}\n```'
    )
    llm = SequenceLLM(["this is not json", good, good, good, good, good, good, good])
    result = _run_ss(llm, compact=True, max_steps=6)
    assert result.steps[0].validation_error
    assert result.steps[0].action == ""
    assert result.steps[0].state_after["step"] == 0  # patch not applied
    assert result.steps[1].validation_error is None
    assert not result.failed


def test_two_validator_failures_abort():
    llm = SequenceLLM(["nope", "still nope", "should not be called"])
    result = _run_ss(llm, compact=True, max_steps=6)
    assert result.failed
    assert len(result.steps) == 2
    assert result.steps[0].state_after == result.steps[1].state_before


def test_history_grows_skillstate_stays_flat():
    skill = WarehouseSkill()
    seed, max_steps = 3, 16
    ss = run_skill_state(
        skill_name="warehouse",
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=skill.make_env(seed, compact=False, horizon=max_steps),
        llm=FakeLLM(warehouse_skillstate_policy),
        parse_action=skill.parse_action,
        max_steps=max_steps,
        seed=seed,
        model="fake",
    )
    hist = run_history(
        skill_name="warehouse",
        instructions=skill.instructions,
        initial_state=skill.initial_state(),
        env=skill.make_env(seed, compact=False, horizon=max_steps),
        llm=FakeLLM(warehouse_history_policy),
        parse_action=skill.parse_action,
        max_steps=max_steps,
        seed=seed,
        model="fake",
    )
    ss_curve = ss.totals.prompt_curve
    hs_curve = hist.totals.prompt_curve
    assert len(ss_curve) >= 8 and len(hs_curve) >= 8
    assert hs_curve[-1] > hs_curve[0] * 1.15
    assert max(ss_curve) <= min(ss_curve) * 1.35 + 80
    assert hs_curve[-1] > ss_curve[-1]
    # History prompt actually contains prior observations.
    assert "History:" in hist.steps[-1].prompt
    assert hist.steps[0].observation in hist.steps[-1].prompt


def test_scripted_policy_emits_two_key_json():
    skill = WarehouseSkill()
    env = skill.make_env(1, compact=True, horizon=8)
    from skillstate.prompts import build_skill_state_prompt

    prompt = build_skill_state_prompt(
        skill.instructions, skill.initial_state(), env.reset()
    )
    text = warehouse_skillstate_policy(prompt)
    obj, _ = extract_json_object(text)
    assert set(obj.keys()) == {"state_patch", "action"}
    json.dumps(obj)  # serializable
