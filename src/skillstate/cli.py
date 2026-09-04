"""Rich CLI: warehouse, repoops, compare, ui."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillstate import __version__
from skillstate.compare import run_compare
from skillstate.fake_llm import FakeLLM
from skillstate.logging_util import EpisodeResult, compact_state, write_run
from skillstate.ollama_client import (
    DEFAULT_MODEL,
    OllamaClient,
    default_base_url,
    default_model,
    health_check,
)
from skillstate.policies import repoops_skillstate_policy, warehouse_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills import load_skill

load_dotenv()

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skillstate",
        description="SKILL.state local runtime (independent PoC of arXiv:2608.26263).",
    )
    parser.add_argument("--version", action="version", version=f"skillstate {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_run_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--max-steps", type=int, default=36)
        p.add_argument("--model", default=None, help=f"Ollama model (default {DEFAULT_MODEL})")
        p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
        p.add_argument(
            "--offline",
            action="store_true",
            help="Use the scripted policy instead of Ollama (for tests/demos).",
        )
        p.add_argument("--compact", action="store_true", help="Short 5-event warehouse episode.")
        p.add_argument(
            "--drift-at",
            type=int,
            default=None,
            metavar="STEP",
            help="Warehouse only: silent MOVE after this step index (off by default).",
        )

    wh = sub.add_parser("warehouse", help="Run the 24-shelf warehouse skill")
    add_run_flags(wh)

    repo = sub.add_parser("repoops", help="Run the toy git/CI skill")
    add_run_flags(repo)

    cmp_ = sub.add_parser("compare", help="SKILL.state vs ReAct-history on the same seed")
    add_run_flags(cmp_)
    cmp_.set_defaults(max_steps=30)

    sweep = sub.add_parser("sweep", help="Same warehouse+drift protocol across local Ollama models")
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--max-steps", type=int, default=40)
    sweep.add_argument("--drift-at", type=int, default=10)
    sweep.add_argument("--model", dest="models", action="append", default=None)
    sweep.add_argument("--base-url", default=None)
    sweep.add_argument("--offline", action="store_true")

    ui = sub.add_parser("ui", help="Serve the local dashboard at http://127.0.0.1:8000")
    ui.add_argument("--host", default=os.environ.get("SKILLSTATE_HOST", "127.0.0.1"))
    ui.add_argument("--port", type=int, default=int(os.environ.get("SKILLSTATE_PORT", "8000")))
    ui.add_argument("--model", default=None)
    ui.add_argument("--base-url", default=None)

    args = parser.parse_args(argv)
    if args.cmd in {"warehouse", "repoops"}:
        return _run_skill(args.cmd, args)
    if args.cmd == "compare":
        model = args.model or default_model()
        if not args.offline:
            health_check(args.base_url or default_base_url(), model)
        run_compare(
            seed=args.seed,
            max_steps=args.max_steps,
            model=model,
            base_url=args.base_url,
            offline=args.offline,
            compact=getattr(args, "compact", False),
            drift_at=getattr(args, "drift_at", None),
            console=console,
        )
        return 0
    if args.cmd == "sweep":
        from skillstate.sweep import run_sweep

        return run_sweep(
            models=args.models,
            seed=args.seed,
            max_steps=args.max_steps,
            drift_at=args.drift_at,
            offline=args.offline,
            base_url=args.base_url,
            console=console,
        )
    if args.cmd == "ui":
        return _run_ui(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


def _run_skill(name: str, args: argparse.Namespace) -> int:
    skill = load_skill(name)
    model = args.model or default_model()
    if args.offline:
        policy = (
            warehouse_skillstate_policy if name == "warehouse" else repoops_skillstate_policy
        )
        llm: Any = FakeLLM(policy, model="fake")
    else:
        health_check(args.base_url or default_base_url(), model)
        llm = OllamaClient(model=model, base_url=args.base_url)

    env_kwargs: dict[str, Any] = {"horizon": args.max_steps}
    if name == "warehouse":
        env_kwargs["compact"] = bool(args.compact)
        if getattr(args, "drift_at", None) is not None:
            env_kwargs["drift_at"] = args.drift_at
    env = skill.make_env(args.seed, **env_kwargs)

    console.rule(f"[bold]SKILL.state[/bold]  skill={name}  model={getattr(llm, 'model', model)}")
    console.print(
        "[dim]A_t = (P, Σ_t, O_t) only. Reasoning is discarded after each step. "
        "Paper: arXiv:2608.26263[/dim]"
    )

    stream_buf: list[str] = []

    def on_token(piece: str) -> None:
        stream_buf.append(piece)

    with Live(
        Panel("", title="R_t (discarded after this step)"),
        console=console,
        refresh_per_second=12,
    ) as live:

        def on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "token":
                live.update(
                    Panel(
                        Text("".join(stream_buf)[-1200:]),
                        title="R_t (streaming, then discarded)",
                    )
                )
            elif event.get("type") == "prompt":
                stream_buf.clear()

        result = run_skill_state(
            skill_name=skill.name,
            instructions=skill.instructions,
            state_schema=skill.state_schema,
            initial_state=skill.initial_state(),
            env=env,
            llm=llm,
            parse_action=skill.parse_action,
            max_steps=args.max_steps,
            seed=args.seed,
            model=getattr(llm, "model", model),
            on_token=on_token,
            on_event=on_event,
        )

    table = Table(expand=True, show_lines=True, pad_edge=False)
    table.add_column("step", justify="right", style="bold", no_wrap=True)
    table.add_column("prompt tok", justify="right", no_wrap=True)
    table.add_column("Σ", overflow="fold", min_width=28)
    table.add_column("O_t", overflow="fold", min_width=24)
    table.add_column("action", style="cyan", min_width=16, overflow="fold")
    table.add_column("ΔΣ", overflow="fold", min_width=20)
    table.add_column("totals", no_wrap=True)
    for step in result.steps:
        obs = step.observation.replace("\n", " | ")
        if len(obs) > 160:
            obs = obs[:157] + "…"
        if step.validation_error:
            obs = f"[red]VALIDATOR[/red] {step.validation_error} | {obs}"
        table.add_row(
            str(step.step),
            str(step.prompt_tokens),
            compact_state(step.state_after),
            obs,
            step.action or "—",
            json.dumps(step.state_patch, ensure_ascii=False) if step.state_patch else "—",
            f"in={step.totals['prompt_tokens']} out={step.totals['completion_tokens']} tot={step.totals['total_tokens']}",
        )
    console.print(table)
    _print_summary(result)
    path = write_run({"cli": name, **result.as_dict()})
    console.print(f"[dim]trace: {path}[/dim]")
    return 0 if result.success else 1


def _print_summary(result: EpisodeResult) -> None:
    t = result.totals
    status = "[green]SUCCESS[/green]" if result.success else "[red]FAILED[/red]"
    if result.fail_reason:
        status += f"  ({result.fail_reason})"
    console.print(
        f"{status}  steps={t.steps}  avg_prompt={t.avg_prompt_tokens:.1f}  "
        f"max_prompt={t.max_prompt_tokens}  total_tokens={t.total}"
    )
    console.print(f"prompt curve: {t.prompt_curve}")
    clf = result.extra.get("classifier") or {}
    if clf:
        console.print(
            "classifier: "
            f"reached_drift={clf.get('reached_drift')} "
            f"grammar_fail_count={clf.get('grammar_fail_count')} "
            f"delete_only={clf.get('delete_only')} "
            f"stale_location={clf.get('stale_location')} "
            f"correct_relocation={clf.get('correct_relocation')} "
            f"starved_correct={clf.get('starved_correct')}"
        )
    if result.extra.get("drift_step") is not None:
        console.print(
            "silent drift: "
            f"step={result.extra.get('drift_step')} "
            f"item={result.extra.get('drifted_item')} "
            f"{result.extra.get('from_shelf')}→{result.extra.get('to_shelf')} "
            f"recovery_lag={result.extra.get('recovery_lag')}"
        )


def _run_ui(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ.setdefault("OLLAMA_MODEL", args.model or default_model())
    if args.base_url:
        os.environ["OLLAMA_BASE_URL"] = args.base_url
    console.print(
        f"SKILL.state UI → http://{args.host}:{args.port}  "
        f"(model={os.environ['OLLAMA_MODEL']})"
    )
    uvicorn.run(
        "skillstate.ui.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
