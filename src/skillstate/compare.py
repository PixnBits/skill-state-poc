"""SKILL.state vs honest ReAct-history on the same warehouse seed."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from skillstate.fake_llm import FakeLLM
from skillstate.history_runtime import run_history
from skillstate.logging_util import EpisodeResult, write_run
from skillstate.ollama_client import OllamaClient
from skillstate.policies import warehouse_history_policy, warehouse_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills.warehouse import WarehouseSkill


def run_compare(
    *,
    seed: int = 42,
    max_steps: int = 30,
    model: str = "",
    base_url: str | None = None,
    offline: bool = False,
    compact: bool = False,
    drift_at: int | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    skill = WarehouseSkill()
    out = console or Console()

    def make_llm(kind: str) -> Any:
        if offline:
            if kind == "skillstate":
                return FakeLLM(warehouse_skillstate_policy, model="fake")
            return FakeLLM(warehouse_history_policy, model="fake")
        return OllamaClient(model=model or None, base_url=base_url)

    ss_llm = make_llm("skillstate")
    env_kwargs: dict[str, Any] = {"compact": compact, "horizon": max_steps}
    if drift_at is not None:
        env_kwargs["drift_at"] = drift_at
    ss_env = skill.make_env(seed, **env_kwargs)
    skillstate = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=ss_env,
        llm=ss_llm,
        parse_action=skill.parse_action,
        max_steps=max_steps,
        seed=seed,
        model=getattr(ss_llm, "model", model),
    )

    hist_llm = make_llm("history")
    hist_env = skill.make_env(seed, **env_kwargs)
    history = run_history(
        skill_name=skill.name,
        instructions=skill.instructions,
        initial_state=skill.initial_state(),
        env=hist_env,
        llm=hist_llm,
        parse_action=skill.parse_action,
        max_steps=max_steps,
        seed=seed,
        model=getattr(hist_llm, "model", model),
    )

    payload = {
        "seed": seed,
        "max_steps": max_steps,
        "offline": offline,
        "drift_at": drift_at,
        "skillstate": skillstate.as_dict(),
        "history": history.as_dict(),
        "proof": _proof(skillstate, history),
    }
    path = write_run(payload)
    payload["path"] = str(path)

    table = Table(title=f"SKILL.state vs history  seed={seed}  T≤{max_steps}")
    table.add_column("runtime")
    table.add_column("success", justify="center")
    table.add_column("avg prompt tokens", justify="right")
    table.add_column("max prompt tokens", justify="right")
    table.add_column("total tokens", justify="right")
    table.add_column("steps", justify="right")
    for name, ep in (("skillstate", skillstate), ("history", history)):
        t = ep.totals
        table.add_row(
            name,
            "yes" if ep.success else "no",
            f"{t.avg_prompt_tokens:.1f}",
            str(t.max_prompt_tokens),
            str(t.total),
            str(t.steps),
        )
    out.print(table)
    out.print(f"[dim]wrote {path}[/dim]")
    proof = payload["proof"]
    if proof["ok"]:
        out.print(
            "[green]Proof holds:[/green] SKILL.state prompt is roughly flat; "
            "history prompt grows."
        )
    else:
        out.print(f"[red]Proof FAILED:[/red] {proof['reason']}")
        out.print(
            "If history is not climbing, the history runtime is not appending "
            "the transcript. If SKILL.state is climbing, something is leaking "
            "prior observations/CoT into A_t."
        )
    out.print("prompt-token curve:")
    out.print(f"  skillstate: {skillstate.totals.prompt_curve}")
    out.print(f"  history:    {history.totals.prompt_curve}")
    if skillstate.extra.get("drift_step") is not None:
        out.print(
            "silent drift recovery_lag="
            f"{skillstate.extra.get('recovery_lag')} "
            f"(step={skillstate.extra.get('drift_step')} "
            f"{skillstate.extra.get('drifted_item')} "
            f"{skillstate.extra.get('from_shelf')}→"
            f"{skillstate.extra.get('to_shelf')})"
        )
    return payload


def _proof(skillstate: EpisodeResult, history: EpisodeResult) -> dict[str, Any]:
    ss = skillstate.totals.prompt_curve
    hs = history.totals.prompt_curve
    if len(ss) < 3 or len(hs) < 3:
        return {"ok": False, "reason": "not enough steps to compare curves"}
    # History must grow: last prompt tokens > first by a clear margin.
    hist_grew = hs[-1] > hs[0] * 1.15 and hs[-1] > hs[0] + 40
    # SKILL.state must stay roughly flat: max within 35% of min (state can
    # grow by a few inbound/pending items, never by the transcript).
    ss_min, ss_max = min(ss), max(ss)
    flat = ss_max <= ss_min * 1.35 + 80
    # Growth *rate* is the claim. At short T, history can still be smaller in
    # absolute tokens because it does not serialize the 24-shelf Σ; it must
    # nevertheless be climbing while SKILL.state is not.
    ss_growth = ss[-1] / max(ss[0], 1)
    hs_growth = hs[-1] / max(hs[0], 1)
    faster = hs_growth > ss_growth * 1.1
    ok = hist_grew and flat and faster
    reason_parts = []
    if not hist_grew:
        reason_parts.append(f"history did not grow ({hs[0]} → {hs[-1]})")
    if not flat:
        reason_parts.append(f"SKILL.state not flat (min={ss_min} max={ss_max})")
    if not faster:
        reason_parts.append(
            f"history growth {hs_growth:.2f}x not ahead of SKILL.state {ss_growth:.2f}x"
        )
    return {
        "ok": ok,
        "reason": "; ".join(reason_parts) if reason_parts else "ok",
        "skillstate_curve": ss,
        "history_curve": hs,
    }
