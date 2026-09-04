"""Canned-patch classifier tests. No Ollama."""

from __future__ import annotations

from skillstate.classify import (
    CORRECT_RELOCATION,
    DELETE_ONLY,
    GRAMMAR_FAIL,
    STALE_LOCATION,
    classify_patch,
    is_starved_correct,
    item_shelf,
    markdown_table,
    summarize_episode,
)
from skillstate.logging_util import EpisodeResult, StepResult, TokenTotals
from skillstate.runtime import run_skill_state
from skillstate.fake_llm import FakeLLM, SequenceLLM
from skillstate.policies import warehouse_skillstate_policy
from skillstate.skills.warehouse import WarehouseSkill


def _inv(**occupied: str | None) -> dict[str, str | None]:
    from skillstate.skills.warehouse.schema import empty_inventory

    inventory = empty_inventory()
    inventory.update(occupied)
    return inventory


def test_item_shelf():
    assert item_shelf(_inv(shelf_01="item_00"), "item_00") == "shelf_01"
    assert item_shelf(_inv(shelf_00="item_00"), "item_99") is None


def test_correct_relocation():
    assert (
        classify_patch(
            proposed_inventory=_inv(shelf_00=None, shelf_01="item_00"),
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
        )
        == CORRECT_RELOCATION
    )


def test_ship_from_scanner_shelf_is_correct_relocation():
    assert (
        classify_patch(
            proposed_inventory=_inv(shelf_00=None, shelf_01=None),
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
            action="SHIP item_00 shelf_01",
        )
        == CORRECT_RELOCATION
    )


def test_delete_only():
    assert (
        classify_patch(
            proposed_inventory=_inv(shelf_00=None, shelf_01=None),
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
        )
        == DELETE_ONLY
    )


def test_stale_location():
    assert (
        classify_patch(
            proposed_inventory=_inv(shelf_00="item_00"),
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
        )
        == STALE_LOCATION
    )


def test_wrong_other_shelf_is_stale():
    assert (
        classify_patch(
            proposed_inventory=_inv(shelf_03="item_00"),
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
        )
        == STALE_LOCATION
    )


def test_grammar_fail():
    assert (
        classify_patch(
            proposed_inventory=None,
            item="item_00",
            from_shelf="shelf_00",
            to_shelf="shelf_01",
            grammar_fail=True,
        )
        == GRAMMAR_FAIL
    )


def test_starved_correct_is_right_patch_and_env_reject():
    assert is_starved_correct(CORRECT_RELOCATION, True)
    assert not is_starved_correct(CORRECT_RELOCATION, False)
    assert not is_starved_correct(DELETE_ONLY, True)


def _step(**kwargs) -> StepResult:
    base = dict(
        step=0,
        prompt="",
        prompt_tokens=10,
        completion_tokens=10,
        observation="",
        reasoning="",
        state_patch={},
        action="WAIT",
        state_before={},
        state_after={},
        changed_keys=[],
        next_observation="",
        validation_error=None,
        env_error=False,
        done=False,
        success=False,
        totals={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    base.update(kwargs)
    return StepResult(**base)


def test_summarize_counts_post_drift_only():
    steps = [
        _step(step=0, patch_class=STALE_LOCATION),
        _step(step=1, patch_class=STALE_LOCATION),  # drift_step
        _step(step=2, patch_class=DELETE_ONLY, env_error=True, action="SHIP item_00 shelf_00"),
        _step(step=3, patch_class=CORRECT_RELOCATION, env_error=True, action="WAIT"),
        _step(step=4, validation_error="bad json", patch_class=GRAMMAR_FAIL),
    ]
    result = EpisodeResult(
        runtime="skillstate",
        skill="warehouse",
        seed=7,
        model="fake",
        success=False,
        failed=False,
        fail_reason=None,
        steps=steps,
        totals=TokenTotals(),
        extra={
            "drift_step": 1,
            "drifted_item": "item_00",
            "from_shelf": "shelf_00",
            "to_shelf": "shelf_01",
            "recovery_lag": None,
        },
    )
    for _ in steps:
        result.totals.add(100, 10)
    row = summarize_episode(result)
    assert row["reached_drift"] is True
    assert row["delete_only"] == 1
    assert row["correct_relocation"] == 1
    assert row["starved_correct"] == 1
    assert row["stale_location"] == 0  # pre-drift stale not counted
    assert row["grammar_fail_count"] == 1
    assert row["recovery_lag"] is None


def test_markdown_table_renders_nulls():
    text = markdown_table(
        [{"model": "x", "reached_drift": False, "steps": 4, "fail_reason": None, "recovery_lag": None,
          "grammar_fail_count": 2, "delete_only": 0, "stale_location": 0, "correct_relocation": 0,
          "starved_correct": 0}]
    )
    assert "null" in text
    assert "no" in text


def test_runtime_records_proposed_shelf_on_rejected_ship():
    """delete_only + rejected SHIP after scanner: proposed shelf persisted, Σ unchanged."""
    skill = WarehouseSkill()
    env = skill.make_env(1, compact=True, horizon=8, drift_at=1)
    store0 = (
        '```json\n{"state_patch": {"inventory": {"shelf_00": "item_00"}, "inbound": [], '
        '"last_action": "STORE item_00 shelf_00", "step": 1}, "action": "STORE item_00 shelf_00"}\n```'
    )
    store1 = (
        '```json\n{"state_patch": {"inventory": {"shelf_01": "item_01"}, "inbound": [], '
        '"last_action": "STORE item_01 shelf_01", "step": 2}, "action": "STORE item_01 shelf_01"}\n```'
    )
    # After drift, scanner names to_shelf; 14B-style patch only nulls from_shelf.
    delete_ship = (
        '```json\n{"state_patch": {"inventory": {"shelf_00": null}, '
        '"last_action": "SHIP item_00 shelf_00", "step": 3}, "action": "SHIP item_00 shelf_00"}\n```'
    )
    llm = SequenceLLM([store0, store1, delete_ship])
    result = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=env,
        llm=llm,
        parse_action=skill.parse_action,
        max_steps=3,
        seed=1,
        model="fake",
    )
    assert result.extra.get("drift_step") == 1
    last = result.steps[2]
    assert last.env_error
    assert last.action.startswith("SHIP")
    assert last.patch_class == DELETE_ONLY
    assert last.proposed_item_shelf is None
    assert last.state_after == last.state_before
    clf = result.extra["classifier"]
    assert clf["delete_only"] >= 1
    assert clf["starved_correct"] == 0


def test_scripted_offline_recovery_is_correct_relocation():
    skill = WarehouseSkill()
    env = skill.make_env(1, compact=True, horizon=8, drift_at=1)
    result = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=env,
        llm=FakeLLM(warehouse_skillstate_policy),
        parse_action=skill.parse_action,
        max_steps=8,
        seed=1,
        model="fake",
    )
    assert result.extra.get("recovery_lag") == 1
    post = [s for s in result.steps if s.step > result.extra["drift_step"]]
    assert any(s.patch_class == CORRECT_RELOCATION for s in post)
