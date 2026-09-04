"""Offline bench harness tests. No Ollama."""

from __future__ import annotations

import json

from skillstate.bench import CELL_KEYS, render_bench_md, run_bench
from skillstate.fake_llm import FakeLLM
from skillstate.history_runtime import run_history
from skillstate.policies import warehouse_history_policy, warehouse_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills.warehouse import WarehouseSkill
from skillstate.sweep import run_warehouse_episode


def test_bench_json_schema_and_wall_s(tmp_path):
    payload = run_bench(
        models=["fake"],
        seeds=[7, 21],
        max_steps=8,
        drift_at=1,
        offline=True,
        compact=True,
        out_dir=tmp_path / "bench",
    )
    cells = payload["cells"]
    assert len(cells) == 4  # 1 model × 2 runtimes × 2 seeds; does not stop after recovery
    runtimes = {(c["runtime"], c["seed"]) for c in cells}
    assert runtimes == {
        ("skillstate", 7),
        ("history", 7),
        ("skillstate", 21),
        ("history", 21),
    }
    assert [c["runtime"] for c in cells[:2]] == ["skillstate", "history"]
    assert cells[0]["cold"] is True
    assert cells[1]["cold"] is False
    for cell in cells:
        for key in CELL_KEYS:
            assert key in cell, f"missing {key}"
        assert cell["wall_s"] >= 0
        assert cell["avg_step_s"] >= 0
        assert isinstance(cell["prompt_curve"], list)
        assert cell["model"] == "offline-scripted"
        path = tmp_path / "bench" / f"offline-scripted__{cell['runtime']}__seed{cell['seed']}.json"
        assert path.is_file()
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["cell"]["wall_s"] >= 0
        assert "episode" in blob
        for step in blob["episode"]["steps"]:
            assert step["wall_s"] >= 0
    hist = next(c for c in cells if c["runtime"] == "history" and c["seed"] == 7)
    ss = next(c for c in cells if c["runtime"] == "skillstate" and c["seed"] == 7)
    assert hist["recovery_via"] == "env_gt_via_actions"
    assert ss["recovery_via"] == "sigma_match"
    assert hist["delete_only"] is None
    assert hist["stale_location"] is None
    assert hist["correct_relocation"] is None
    assert hist["starved_correct"] is None
    assert ss["delete_only"] is not None
    matrix = json.loads((tmp_path / "bench" / "matrix.json").read_text(encoding="utf-8"))
    assert "cells" in matrix
    md = (tmp_path / "BENCH.md").read_text(encoding="utf-8")
    assert "## Table 1" in md
    assert "## Table 2" in md
    assert "Caveats" in md
    assert "offline-scripted" in md
    assert "not a live Table 2 row" in md.lower() or "harness check" in md.lower()
    assert "*not run*" in md  # Table 2 never treats offline-scripted as a live pair


def test_bench_resume_skips_existing(tmp_path):
    first = run_bench(
        models=["fake"],
        seeds=[7],
        max_steps=8,
        drift_at=1,
        offline=True,
        compact=True,
        out_dir=tmp_path / "bench",
    )
    walls = [c["wall_s"] for c in first["cells"]]
    second = run_bench(
        models=["fake"],
        seeds=[7],
        max_steps=8,
        drift_at=1,
        offline=True,
        compact=True,
        out_dir=tmp_path / "bench",
        force=False,
    )
    assert [c["wall_s"] for c in second["cells"]] == walls
    assert len(second["cells"]) == 2


def test_render_partial_banner():
    md = render_bench_md({"partial": True, "cells": [], "commit": "deadbeef", "max_steps": 40, "drift_at": 10, "seeds": [7]})
    assert "**PARTIAL**" in md
    assert "deadbeef" in md


def test_history_recovery_is_env_gt_via_actions_not_location():
    skill = WarehouseSkill()
    env = skill.make_env(1, compact=True, horizon=8, drift_at=1)
    result = run_history(
        skill_name=skill.name,
        instructions=skill.instructions,
        initial_state=skill.initial_state(),
        env=env,
        llm=FakeLLM(warehouse_history_policy),
        parse_action=skill.parse_action,
        max_steps=8,
        seed=1,
        model="fake",
    )
    assert result.extra.get("recovery_via") == "env_gt_via_actions"
    assert result.extra.get("drift_step") == 1
    assert result.extra.get("recovery_lag") is not None
    assert result.extra["recovery_lag"] >= 1
    drift_step = result.steps[1]
    rec = env.silent_drift_record
    assert rec is not None
    item = rec["drifted_item"]
    to_shelf = rec["to_shelf"]
    # Immediately after the drift step the env already has the item on to_shelf.
    assert drift_step.state_after["inventory"].get(to_shelf) == item or item in (
        drift_step.state_after.get("shipped") or []
    )
    # Recovery is not that location: it is a later valid action naming to_shelf, or shipped.
    recovered = result.steps[int(rec["drift_step"]) + int(result.extra["recovery_lag"])]
    targeted = (
        recovered.action.startswith(f"SHIP {item} {to_shelf}")
        or recovered.action.startswith(f"MOVE {item} ")
        and recovered.action.endswith(to_shelf)
        or item in (recovered.state_after.get("shipped") or [])
    )
    assert targeted
    assert recovered.wall_s >= 0
    assert result.extra.get("wall_s", 0) >= 0


def test_skillstate_episode_records_wall_s():
    skill = WarehouseSkill()
    result = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=skill.make_env(1, compact=True, horizon=8),
        llm=FakeLLM(warehouse_skillstate_policy),
        parse_action=skill.parse_action,
        max_steps=8,
        seed=1,
        model="fake",
    )
    assert result.extra.get("wall_s", -1) >= 0
    assert result.steps
    assert all(s.wall_s >= 0 for s in result.steps)


def test_run_warehouse_episode_history_offline():
    result = run_warehouse_episode(
        model="fake",
        seed=1,
        max_steps=8,
        drift_at=1,
        offline=True,
        runtime="history",
        compact=True,
    )
    assert result.runtime == "history"
    assert result.extra.get("wall_s", -1) >= 0
    assert "patch_class" not in (result.extra or {}) or result.extra.get("recovery_via") == "env_gt_via_actions"
