"""Same warehouse + silent-drift protocol across local Ollama models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from skillstate.classify import markdown_table, summarize_episode
from skillstate.fake_llm import FakeLLM
from skillstate.logging_util import EpisodeResult, write_run
from skillstate.ollama_client import OllamaClient, default_base_url, health_check
from skillstate.policies import warehouse_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills.warehouse import WarehouseSkill

# Ordered live candidates. muse-glimmer is excluded. Do not start 35B / q8
# until a 27B/31B finishes an episode.
PRIMARY = ["qwen3.8:27b", "gemma4:31b", "granite4.2:30b"]
FALLBACK = [
    "gemma4:26b",
    "qwen3-coder:30b",
    "nemotron3:33b",
    "devstral-small-2:latest",
    "qwen3.8:27b-q8_0",
    "qwen3.6:35b",
    "qwen2.5:14b",
]
SKIP = {"muse-glimmer:latest", "muse-glimmer"}


def run_warehouse_episode(
    *,
    model: str,
    seed: int,
    max_steps: int,
    drift_at: int | None,
    offline: bool,
    base_url: str | None = None,
) -> EpisodeResult:
    skill = WarehouseSkill()
    env_kwargs: dict[str, Any] = {"horizon": max_steps}
    if drift_at is not None:
        env_kwargs["drift_at"] = drift_at
    env = skill.make_env(seed, **env_kwargs)
    if offline:
        llm: Any = FakeLLM(warehouse_skillstate_policy, model="fake")
    else:
        health_check(base_url or default_base_url(), model)
        llm = OllamaClient(model=model, base_url=base_url)
    try:
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
            model=getattr(llm, "model", model),
        )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()


def _grammar_abort_before_drift(row: dict[str, Any]) -> bool:
    return (not row.get("reached_drift")) and int(row.get("grammar_fail_count") or 0) >= 2


def _recovered(row: dict[str, Any]) -> bool:
    return row.get("recovery_lag") is not None and not isinstance(row.get("recovery_lag"), bool)


def select_models(explicit: list[str] | None) -> list[str]:
    if explicit:
        return [m for m in explicit if m not in SKIP]
    return list(PRIMARY)


def run_sweep(
    *,
    models: list[str] | None = None,
    seed: int = 7,
    max_steps: int = 40,
    drift_at: int = 10,
    offline: bool = False,
    base_url: str | None = None,
    console: Console | None = None,
) -> int:
    out = console or Console()
    planned = select_models(models)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    explicit = bool(models)

    def _run(model: str, *, is_offline: bool) -> dict[str, Any]:
        label = "fake" if is_offline else model
        out.rule(f"sweep  model={label}  seed={seed}  drift_at={drift_at}  T≤{max_steps}")
        result = run_warehouse_episode(
            model=model,
            seed=seed,
            max_steps=max_steps,
            drift_at=drift_at,
            offline=is_offline,
            base_url=base_url,
        )
        row = summarize_episode(result)
        if is_offline:
            row["model"] = "offline-scripted"
        if not row.get("prompt_flat"):
            out.print(
                "[red]Prompt curve is not flat. SKILL.state prompt builder is broken; "
                "stopping the sweep.[/red]"
            )
            out.print(f"curve: {row.get('prompt_curve')}")
        traces.append({"model": row["model"], "summary": row, "episode": result.as_dict()})
        rows.append(row)
        _print_row(out, row)
        return row

    if not explicit:
        control = _run("fake", is_offline=True)
        if control.get("recovery_lag") != 1:
            out.print(
                f"[red]offline control recovery_lag={control.get('recovery_lag')}, "
                "expected 1. Harness is not known-good; not starting live models.[/red]"
            )
            _write_outputs(rows, traces, seed, max_steps, drift_at, out)
            return 1
        if offline:
            _write_outputs(rows, traces, seed, max_steps, drift_at, out)
            return 0

    queue = list(planned)
    if not explicit:
        queue = list(PRIMARY)

    i = 0
    while i < len(queue):
        model = queue[i]
        row = _run(model, is_offline=offline)
        if not row.get("prompt_flat"):
            _write_outputs(rows, traces, seed, max_steps, drift_at, out)
            return 2
        if _recovered(row):
            out.print(f"[green]{model} recovered (recovery_lag={row['recovery_lag']}). Stopping.[/green]")
            break
        i += 1
        if not models and i == len(PRIMARY):
            if all(_grammar_abort_before_drift(r) for r in rows if r["model"] != "offline-scripted"):
                out.print(
                    "First three all grammar-aborted before drift; continuing fallback list."
                )
                queue.extend(m for m in FALLBACK if m not in queue)
            else:
                out.print("No recovery from the first three; not sweeping the rest.")
                break

    _write_outputs(rows, traces, seed, max_steps, drift_at, out)
    return 0 if any(_recovered(r) for r in rows) else 1


def _print_row(out: Console, row: dict[str, Any]) -> None:
    table = Table(show_header=True)
    for col in (
        "model",
        "reached_drift",
        "steps",
        "recovery_lag",
        "grammar_fail_count",
        "delete_only",
        "stale_location",
        "correct_relocation",
        "starved_correct",
    ):
        table.add_column(col)
    table.add_row(
        str(row.get("model")),
        "yes" if row.get("reached_drift") else "no",
        str(row.get("steps")),
        str(row.get("recovery_lag")),
        str(row.get("grammar_fail_count")),
        str(row.get("delete_only")),
        str(row.get("stale_location")),
        str(row.get("correct_relocation")),
        str(row.get("starved_correct")),
    )
    out.print(table)
    if row.get("fail_reason"):
        out.print(f"fail_reason: {row['fail_reason']}")


def _write_outputs(
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    seed: int,
    max_steps: int,
    drift_at: int,
    out: Console,
) -> None:
    payload = {
        "runtime": "sweep",
        "seed": seed,
        "max_steps": max_steps,
        "drift_at": drift_at,
        "rows": rows,
        "traces": traces,
    }
    path = write_run(payload, suffix="sweep")
    notes = Path.cwd() / "runs" / "MODEL-NOTES.md"
    notes.write_text(
        _notes_markdown(rows, traces, seed, max_steps, drift_at, path),
        encoding="utf-8",
    )
    out.print(f"[dim]wrote {path}[/dim]")
    out.print(f"[dim]wrote {notes}[/dim]")


def _notes_markdown(
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    seed: int,
    max_steps: int,
    drift_at: int,
    json_path: Path,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Model sweep notes",
        "",
        f"Generated {stamp}. Protocol: warehouse seed={seed} max-steps={max_steps} "
        f"drift-at={drift_at}, temperature 0.0, SKILL.state runtime (no second loop).",
        "",
        "P (`skills/warehouse/skill.md`) already says cycle-count / scanner "
        "observations override Σ (`If O contradicts Σ, patch inventory to match O`). "
        "No extra “trust the scanner” sentence was added.",
        "",
        "Commit rule unchanged: proposed Σ is classified even when `env.step` rejects; "
        "live Σ is still committed only on a valid action.",
        "",
        "## Table",
        "",
        markdown_table(rows),
        "",
        f"Raw JSON: `{json_path.name}` (gitignored).",
        "",
        "## First post-scanner action (quoted from traces)",
        "",
    ]
    for row in rows:
        fps = row.get("first_post_scanner")
        parts.append(f"### {row.get('model')}")
        parts.append("")
        parts.append(
            f"reached_drift={row.get('reached_drift')}  recovery_lag={row.get('recovery_lag')}  "
            f"fail_reason={row.get('fail_reason')!r}  prompt_flat={row.get('prompt_flat')}"
        )
        parts.append("")
        if not fps:
            parts.append("No cycle-count observation in this episode (grammar abort or no drift).")
            parts.append("")
            continue
        parts.append(f"step {fps.get('step')}, patch_class=`{fps.get('patch_class')}`, "
                     f"proposed_item_shelf=`{fps.get('proposed_item_shelf')}`, "
                     f"env_error={fps.get('env_error')}")
        parts.append("")
        parts.append("Observation:")
        parts.append("")
        parts.append("```")
        parts.append((fps.get("observation") or "").strip())
        parts.append("```")
        parts.append("")
        parts.append(f"action: `{fps.get('action')}`")
        parts.append("")
        parts.append("state_patch:")
        parts.append("")
        parts.append("```json")
        import json

        parts.append(json.dumps(fps.get("state_patch") or {}, indent=2, ensure_ascii=False))
        parts.append("```")
        parts.append("")
        label = fps.get("patch_class")
        if label == "correct_relocation" and fps.get("env_error"):
            parts.append("Classification: **starved_correct** (right patch, env rejected the action).")
        elif label == "correct_relocation":
            parts.append("Classification: **correct_relocation**.")
        elif label == "delete_only":
            parts.append("Classification: **delete_only** (scanner-as-delete; item on neither shelf).")
        elif label == "stale_location":
            parts.append("Classification: **stale_location** (item still on from_shelf).")
        elif label == "grammar_fail":
            parts.append("Classification: **grammar_fail**.")
        else:
            parts.append(f"Classification: `{label}`.")
        parts.append("")
    return "\n".join(parts) + "\n"
