"""SKILL.state vs history matrix on local Ollama models.

Does not stop after the first recovery. Per model the order is
skillstate seed N, history seed N (skillstate first so cold-load cost
is not dumped only onto history).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from skillstate.classify import first_post_scanner_step, summarize_episode
from skillstate.logging_util import EpisodeResult
from skillstate.ollama_client import _model_present, default_base_url, health_check
from skillstate.sweep import run_warehouse_episode

TIER_A = [
    "qwen2.5:14b",
    "qwen3.8:27b",
    "gemma4:26b",
    "gemma4:31b",
    "granite4.2:30b",
]
TIER_B = [
    "gemma4:e4b",
    "qwen3-coder:30b",
    "nemotron3:33b",
    "qwen3.6:35b",
]
SKIP = {"muse-glimmer:latest", "muse-glimmer"}
PARAMS_ISH = {
    "qwen2.5:14b": "14B",
    "qwen3.8:27b": "27B",
    "qwen3.8:27b-q8_0": "27B-q8",
    "gemma4:26b": "26B",
    "gemma4:31b": "31B",
    "gemma4:e4b": "e4B",
    "granite4.2:30b": "30B",
    "qwen3-coder:30b": "30B",
    "nemotron3:33b": "33B",
    "qwen3.6:35b": "35B",
    "fake": "scripted",
    "offline-scripted": "scripted",
}

CELL_KEYS = (
    "model",
    "runtime",
    "seed",
    "success",
    "reached_drift",
    "recovery_lag",
    "fail_reason",
    "steps",
    "wall_s",
    "avg_step_s",
    "avg_prompt_tokens",
    "max_prompt_tokens",
    "total_tokens",
    "completion_tokens",
    "prompt_curve",
    "grammar_fail_count",
    "delete_only",
    "stale_location",
    "correct_relocation",
    "starved_correct",
    "cold",
    "cell_index",
    "first_post_scanner_targeted_to_shelf",
    "recovery_via",
)

HEADLINE_PAIRS = [
    ("qwen2.5:14b", "skillstate", "qwen3.8:27b", "history"),
    ("qwen3.8:27b", "skillstate", "gemma4:31b", "history"),
    ("gemma4:26b", "skillstate", "gemma4:31b", "history"),
    ("qwen3.8:27b", "skillstate", "qwen3.6:35b", "history"),
]


def slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def cell_path(bench_dir: Path, model: str, runtime: str, seed: int) -> Path:
    return bench_dir / f"{slug(model)}__{runtime}__seed{seed}.json"


def cell_record(
    result: EpisodeResult,
    *,
    runtime: str,
    seed: int,
    model: str,
    cold: bool,
    cell_index: int,
) -> dict[str, Any]:
    summary = summarize_episode(result)
    wall_s = float(result.extra.get("wall_s") or 0.0)
    steps = int(summary.get("steps") or result.totals.steps or 0)
    step_walls = [float(s.wall_s) for s in result.steps if getattr(s, "wall_s", 0)]
    avg_step = (sum(step_walls) / len(step_walls)) if step_walls else 0.0
    scanner = first_post_scanner_step(result.steps)
    targeted = None
    if runtime == "history":
        targeted = result.extra.get("first_post_scanner_targeted_to_shelf")
        summary["delete_only"] = None
        summary["stale_location"] = None
        summary["correct_relocation"] = None
        summary["starved_correct"] = None
    elif scanner:
        to_shelf = str(summary.get("to_shelf") or "")
        targeted = bool(to_shelf) and to_shelf in (scanner.get("action") or "")
    return {
        "model": model,
        "runtime": runtime,
        "seed": seed,
        "success": bool(result.success),
        "reached_drift": bool(summary.get("reached_drift")),
        "recovery_lag": summary.get("recovery_lag"),
        "fail_reason": result.fail_reason,
        "steps": steps,
        "wall_s": round(wall_s, 3),
        "avg_step_s": round(avg_step, 3),
        "avg_prompt_tokens": round(result.totals.avg_prompt_tokens, 1),
        "max_prompt_tokens": result.totals.max_prompt_tokens,
        "total_tokens": result.totals.total,
        "completion_tokens": result.totals.completion_tokens,
        "prompt_curve": list(result.totals.prompt_curve),
        "grammar_fail_count": summary.get("grammar_fail_count"),
        "delete_only": summary.get("delete_only"),
        "stale_location": summary.get("stale_location"),
        "correct_relocation": summary.get("correct_relocation"),
        "starved_correct": summary.get("starved_correct"),
        "cold": cold,
        "cell_index": cell_index,
        "first_post_scanner_targeted_to_shelf": targeted,
        "recovery_via": result.extra.get("recovery_via")
        if runtime == "history"
        else "sigma_match",
        "prompt_flat": summary.get("prompt_flat"),
        "params_ish": PARAMS_ISH.get(model, "?"),
        "first_post_scanner": summary.get("first_post_scanner"),
        "drift_step": summary.get("drift_step"),
        "drifted_item": summary.get("drifted_item"),
        "from_shelf": summary.get("from_shelf"),
        "to_shelf": summary.get("to_shelf"),
        "failed": result.failed,
        "offline": model in {"fake", "offline-scripted"} or result.model == "fake",
    }


def error_cell(
    *,
    model: str,
    runtime: str,
    seed: int,
    cold: bool,
    cell_index: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "runtime": runtime,
        "seed": seed,
        "success": False,
        "reached_drift": False,
        "recovery_lag": None,
        "fail_reason": reason,
        "steps": 0,
        "wall_s": 0.0,
        "avg_step_s": 0.0,
        "avg_prompt_tokens": 0.0,
        "max_prompt_tokens": 0,
        "total_tokens": 0,
        "completion_tokens": 0,
        "prompt_curve": [],
        "grammar_fail_count": 0,
        "delete_only": None if runtime == "history" else 0,
        "stale_location": None if runtime == "history" else 0,
        "correct_relocation": None if runtime == "history" else 0,
        "starved_correct": None if runtime == "history" else 0,
        "cold": cold,
        "cell_index": cell_index,
        "first_post_scanner_targeted_to_shelf": None,
        "recovery_via": "env_gt_via_actions" if runtime == "history" else "sigma_match",
        "prompt_flat": True,
        "params_ish": PARAMS_ISH.get(model, "?"),
        "first_post_scanner": None,
        "drift_step": None,
        "drifted_item": None,
        "from_shelf": None,
        "to_shelf": None,
        "failed": True,
        "offline": model in {"fake", "offline-scripted"},
    }


def parse_models(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return list(TIER_A)
    if isinstance(raw, list):
        items = raw
    else:
        items = [p.strip() for p in raw.split(",") if p.strip()]
    seen: list[str] = []
    for m in items:
        if m in SKIP or m in seen:
            continue
        seen.append(m)
    return seen


def parse_seeds(raw: str | list[int] | None) -> list[int]:
    if raw is None:
        return [7, 21]
    if isinstance(raw, list):
        return [int(s) for s in raw]
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_cell_file(path: Path, row: dict[str, Any], episode: dict[str, Any] | None) -> None:
    path.write_text(
        json.dumps({"cell": row, "episode": episode}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_bench(
    *,
    models: list[str] | str | None = None,
    seeds: list[int] | str | None = None,
    max_steps: int = 40,
    drift_at: int = 10,
    offline: bool = False,
    base_url: str | None = None,
    out_dir: Path | None = None,
    console: Console | None = None,
    include_tier_b: bool = False,
    compact: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    out = console or Console()
    model_list = parse_models(models)
    if include_tier_b:
        model_list = model_list + [m for m in TIER_B if m not in model_list]
    if offline and models is None:
        model_list = ["fake"]
    seed_list = parse_seeds(seeds)
    bench_dir = Path(out_dir) if out_dir is not None else (Path.cwd() / "runs" / "bench")
    bench_dir.mkdir(parents=True, exist_ok=True)
    md_path = bench_dir.parent / "BENCH.md"
    cells: list[dict[str, Any]] = []
    skipped: list[str] = []
    sha = _git_sha()
    started = datetime.now(timezone.utc).isoformat()
    cell_index = 0
    complete = True
    installed: list[str] | None = None

    def persist(partial: bool) -> dict[str, Any]:
        payload = {
            "partial": partial,
            "commit": sha,
            "started": started,
            "updated": datetime.now(timezone.utc).isoformat(),
            "max_steps": max_steps,
            "drift_at": drift_at,
            "seeds": seed_list,
            "models": model_list,
            "offline": offline,
            "compact": compact,
            "cells": cells,
            "skipped": skipped,
            "hardware": "Framework Desktop, Ollama local, temperature 0.0",
        }
        (bench_dir / "matrix.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        md_path.write_text(render_bench_md(payload), encoding="utf-8")
        return payload

    persist(partial=True)

    if not offline:
        try:
            info = health_check(base_url or default_base_url(), require_model=False)
            installed = list(info.get("models") or [])
        except SystemExit:
            complete = False
            persist(partial=True)
            raise

    try:
        for model in model_list:
            if not offline and installed is not None and not _model_present(model, installed):
                out.print(f"[yellow]skip {model}: not installed (will not pull)[/yellow]")
                skipped.append(model)
                persist(partial=True)
                continue
            first_for_model = True
            for seed in seed_list:
                for runtime in ("skillstate", "history"):
                    label = "offline-scripted" if offline else model
                    path = cell_path(bench_dir, label, runtime, seed)
                    if path.exists() and not force:
                        try:
                            existing = json.loads(path.read_text(encoding="utf-8"))
                            row = existing.get("cell") or existing
                            row["cell_index"] = cell_index
                            cells.append(row)
                            first_for_model = False
                            out.print(
                                f"  resume skip {label} {runtime} seed={seed} "
                                f"(success={row.get('success')} wall_s={row.get('wall_s')})"
                            )
                            cell_index += 1
                            persist(partial=True)
                            continue
                        except (OSError, json.JSONDecodeError):
                            pass
                    cold = first_for_model
                    first_for_model = False
                    out.rule(
                        f"bench  {label}  {runtime}  seed={seed}  "
                        f"{'COLD' if cold else 'warm'}  T≤{max_steps} drift_at={drift_at}"
                    )
                    try:
                        result = run_warehouse_episode(
                            model=model if not offline else "fake",
                            seed=seed,
                            max_steps=max_steps,
                            drift_at=drift_at,
                            offline=offline,
                            base_url=base_url,
                            runtime=runtime,
                            compact=compact,
                        )
                        row = cell_record(
                            result,
                            runtime=runtime,
                            seed=seed,
                            model=label,
                            cold=cold,
                            cell_index=cell_index,
                        )
                        episode = result.as_dict()
                    except (Exception, SystemExit) as exc:
                        if isinstance(exc, SystemExit) and exc.code in (0, None):
                            raise
                        complete = False
                        reason = f"{type(exc).__name__}: {exc}"
                        out.print(f"[red]cell failed: {reason}[/red]")
                        row = error_cell(
                            model=label,
                            runtime=runtime,
                            seed=seed,
                            cold=cold,
                            cell_index=cell_index,
                            reason=reason,
                        )
                        episode = None
                    cells.append(row)
                    _write_cell_file(path, row, episode)
                    out.print(
                        f"  success={row['success']} recovered={row['recovery_lag']} "
                        f"steps={row['steps']} wall_s={row['wall_s']} "
                        f"tokens={row['total_tokens']} cold={cold}"
                    )
                    cell_index += 1
                    persist(partial=True)
    except KeyboardInterrupt:
        complete = False
        out.print("[yellow]bench interrupted[/yellow]")
        persist(partial=True)
        raise
    planned = len(model_list) * len(seed_list) * 2
    ran_or_resumed = len(cells)
    skipped_models = len(skipped)
    complete = complete and ran_or_resumed >= (planned - skipped_models * len(seed_list) * 2)
    payload = persist(partial=not complete)
    out.print(f"[dim]wrote {bench_dir / 'matrix.json'}[/dim]")
    out.print(f"[dim]wrote {md_path}[/dim]")
    payload["complete"] = complete
    payload["md_path"] = str(md_path)
    payload["bench_dir"] = str(bench_dir)
    return payload


def _mean_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault((c["model"], c["runtime"]), []).append(c)
    means = []
    for (model, runtime), rows in groups.items():
        if len(rows) < 2:
            continue

        def avg(key: str, rs: list[dict[str, Any]] = rows) -> float | None:
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            if not vals:
                return None
            return sum(vals) / len(vals)

        lags = [r["recovery_lag"] for r in rows if r.get("recovery_lag") is not None]
        means.append(
            {
                "model": model,
                "runtime": runtime,
                "seed": "mean",
                "params_ish": rows[0].get("params_ish"),
                "success": sum(1 for r in rows if r["success"]) / len(rows),
                "recovered": sum(1 for r in rows if r.get("recovery_lag") is not None) / len(rows),
                "recovery_lag": (sum(lags) / len(lags)) if lags else None,
                "steps": avg("steps"),
                "wall_s": avg("wall_s"),
                "avg_prompt_tokens": avg("avg_prompt_tokens"),
                "max_prompt_tokens": avg("max_prompt_tokens"),
                "total_tokens": avg("total_tokens"),
            }
        )
    return means


def _find_cell(
    cells: list[dict[str, Any]], model: str, runtime: str, seed: int | None = None
) -> dict[str, Any] | None:
    hits = [c for c in cells if c["model"] == model and c["runtime"] == runtime]
    if seed is not None:
        hits = [c for c in hits if c["seed"] == seed]
    return hits[0] if hits else None


def _pair_verdict(small_ss: dict[str, Any], big_h: dict[str, Any]) -> dict[str, Any]:
    ss_ok = small_ss.get("success")
    h_ok = big_h.get("success")
    if isinstance(ss_ok, bool):
        ss_ok_n = 1.0 if ss_ok else 0.0
    else:
        ss_ok_n = float(ss_ok or 0.0)
    if isinstance(h_ok, bool):
        h_ok_n = 1.0 if h_ok else 0.0
    else:
        h_ok_n = float(h_ok or 0.0)
    ss_rec = small_ss.get("recovery_lag") is not None
    h_rec = big_h.get("recovery_lag") is not None
    if "recovered" in small_ss and small_ss.get("seed") == "mean":
        ss_rec_n = float(small_ss.get("recovered") or 0.0)
        h_rec_n = float(big_h.get("recovered") or 0.0)
        ss_rec = ss_rec_n >= h_rec_n
        h_rec_ge = h_rec_n
        ss_rec_ge = ss_rec_n
    else:
        ss_rec_n = 1.0 if ss_rec else 0.0
        h_rec_n = 1.0 if h_rec else 0.0
        ss_rec_ge = ss_rec_n
        h_rec_ge = h_rec_n
    wall_ratio = None
    tok_ratio = None
    if big_h.get("wall_s") and small_ss.get("wall_s"):
        wall_ratio = round(float(small_ss["wall_s"]) / float(big_h["wall_s"]), 3)
    if big_h.get("total_tokens") and small_ss.get("total_tokens"):
        tok_ratio = round(float(small_ss["total_tokens"]) / float(big_h["total_tokens"]), 3)
    wins = (
        ss_ok_n >= h_ok_n
        and ss_rec_ge >= h_rec_ge
        and (
            (wall_ratio is not None and wall_ratio < 1)
            or (tok_ratio is not None and tok_ratio < 1)
        )
    )
    return {
        "success_match": ss_ok_n == h_ok_n,
        "ss_success": ss_ok,
        "h_success": h_ok,
        "recovery_match": ss_rec_n == h_rec_n,
        "ss_recovered": bool(ss_rec_n),
        "h_recovered": bool(h_rec_n),
        "wall_s_ratio": wall_ratio,
        "total_token_ratio": tok_ratio,
        "smaller_skillstate_wins": wins,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def render_bench_md(payload: dict[str, Any]) -> str:
    cells: list[dict[str, Any]] = payload.get("cells") or []
    live_cells = [c for c in cells if not c.get("offline") and c.get("model") != "offline-scripted"]
    table_cells = live_cells if live_cells else cells
    partial = payload.get("partial", False)
    lines = [
        "# SKILL.state vs history bench",
        "",
    ]
    if partial:
        lines += [
            "> **PARTIAL** — matrix stopped before every planned cell finished. "
            "Numbers below are from completed cells only.",
            "",
        ]
    skipped = payload.get("skipped") or []
    lines += [
        f"Hardware: {payload.get('hardware') or 'Framework Desktop, Ollama local, temperature 0.0'}. "
        f"Date {payload.get('updated') or payload.get('started')}. "
        f"Commit `{payload.get('commit')}`.",
        "",
        f"Protocol: warehouse `--max-steps {payload.get('max_steps')} "
        f"--drift-at {payload.get('drift_at')}` seeds `{payload.get('seeds')}`. "
        "Per model: skillstate then history on seed 7, then the same on seed 21 "
        "(first cell of a model is cold-load; later cells on that model are warm).",
        "",
        "History has no Σ. `recovery_lag` on history rows is steps after drift "
        "until a *valid* action names `to_shelf` for the drifted item, or the env "
        "has shipped it. We do not invent a history `patch_class`. "
        "`loc == to_shelf` is already true immediately after the silent MOVE, so "
        "it does not count as recovery without an action.",
        "",
        "These are this machine's numbers. Not the paper's 16× token table.",
        "",
    ]
    if skipped:
        lines += [f"Skipped (not installed, not pulled): {', '.join(skipped)}.", ""]
    if payload.get("offline"):
        lines += [
            "This payload is an **offline-scripted** harness check. "
            "It is not a live Table 2 row.",
            "",
        ]
    lines += [
        "## Table 1 — full matrix",
        "",
        "| model | params-ish | runtime | seed | success | recovered | recovery_lag | steps | wall_s | avg prompt | max prompt | total tokens | cold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in table_cells:
        rec = "yes" if c.get("recovery_lag") is not None else "no"
        lag = "null" if c.get("recovery_lag") is None else str(c.get("recovery_lag"))
        lines.append(
            f"| {c.get('model')} | {c.get('params_ish', '?')} | {c.get('runtime')} | "
            f"{c.get('seed')} | {'yes' if c.get('success') else 'no'} | {rec} | {lag} | "
            f"{c.get('steps')} | {c.get('wall_s')} | {c.get('avg_prompt_tokens')} | "
            f"{c.get('max_prompt_tokens')} | {c.get('total_tokens')} | "
            f"{'yes' if c.get('cold') else 'no'} |"
        )
    means = _mean_rows(table_cells)
    for m in means:
        lag = "null" if m.get("recovery_lag") is None else f"{m['recovery_lag']:.2f}"
        wall = f"{m['wall_s']:.1f}" if isinstance(m.get("wall_s"), float) else m.get("wall_s")
        tot = (
            f"{m['total_tokens']:.0f}"
            if isinstance(m.get("total_tokens"), float)
            else m.get("total_tokens")
        )
        avg_p = (
            f"{m['avg_prompt_tokens']:.1f}"
            if isinstance(m.get("avg_prompt_tokens"), float)
            else m.get("avg_prompt_tokens")
        )
        steps = (
            f"{m['steps']:.1f}" if isinstance(m.get("steps"), float) else m.get("steps")
        )
        lines.append(
            f"| {m.get('model')} | {m.get('params_ish', '?')} | {m.get('runtime')} | mean | "
            f"{m.get('success')} | {m.get('recovered')} | {lag} | {steps} | "
            f"{wall} | {avg_p} | {_fmt(m.get('max_prompt_tokens'))} | {tot} | — |"
        )
    lines += ["", "## Table 2 — cross-size headline pairs", ""]
    lines.append(
        "| smaller skillstate | larger history | seeds | success | recovery | wall_s ratio | token ratio | ss wins? |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    pair_cells = live_cells  # never use offline-scripted in Table 2
    for sm, sr, bm, br in HEADLINE_PAIRS:
        ss_cells = [c for c in pair_cells if c["model"] == sm and c["runtime"] == sr]
        h_cells = [c for c in pair_cells if c["model"] == bm and c["runtime"] == br]
        if not ss_cells or not h_cells:
            lines.append(
                f"| {sm} {sr} | {bm} {br} | — | *not run* | — | — | — | — |"
            )
            continue
        seeds_ss = {c["seed"] for c in ss_cells}
        seeds_h = {c["seed"] for c in h_cells}
        common = sorted(s for s in (seeds_ss & seeds_h) if s != "mean")
        if not common:
            lines.append(
                f"| {sm} {sr} | {bm} {br} | — | *no shared seed* | — | — | — | — |"
            )
            continue
        for seed in common:
            ss = _find_cell(pair_cells, sm, sr, seed)
            h = _find_cell(pair_cells, bm, br, seed)
            if not ss or not h:
                continue
            v = _pair_verdict(ss, h)
            lines.append(
                f"| {sm} {sr} | {bm} {br} | {seed} | "
                f"ss={v['ss_success']} h={v['h_success']} match={v['success_match']} | "
                f"ss={v['ss_recovered']} h={v['h_recovered']} match={v['recovery_match']} | "
                f"{v['wall_s_ratio']} | {v['total_token_ratio']} | "
                f"{'yes' if v['smaller_skillstate_wins'] else 'no'} |"
            )
    lines += [
        "",
        "A smaller skillstate model “wins” a pair if success and recovery are "
        "≥ the larger history model and wall_s **or** total tokens are lower.",
        "",
        "## Table 3 — skillstate failure taxonomy",
        "",
        "History rows have no `patch_class`. Quotes below are skillstate only.",
        "",
    ]
    quoted: set[str] = set()
    for c in table_cells:
        if c["runtime"] != "skillstate":
            continue
        if c["model"] in quoted:
            continue
        # Prefer seed 7 when present; otherwise the first skillstate cell.
        siblings = [
            r
            for r in table_cells
            if r["model"] == c["model"] and r["runtime"] == "skillstate"
        ]
        chosen = next((r for r in siblings if r.get("seed") == 7), c)
        if chosen is not c and c.get("seed") != 7:
            continue
        quoted.add(c["model"])
        fps = chosen.get("first_post_scanner")
        lines.append(f"### {chosen['model']} seed={chosen['seed']}")
        lines.append("")
        lines.append(
            f"reached_drift={chosen.get('reached_drift')} recovery_lag={chosen.get('recovery_lag')} "
            f"grammar_fail_count={chosen.get('grammar_fail_count')} "
            f"delete_only={chosen.get('delete_only')} stale_location={chosen.get('stale_location')} "
            f"correct_relocation={chosen.get('correct_relocation')} "
            f"starved_correct={chosen.get('starved_correct')} fail_reason={chosen.get('fail_reason')!r}"
        )
        lines.append("")
        if not fps:
            lines.append("No cycle-count observation (grammar abort before drift, or no drift).")
            lines.append("")
            continue
        lines.append(
            f"step {fps.get('step')} patch_class=`{fps.get('patch_class')}` "
            f"proposed_item_shelf=`{fps.get('proposed_item_shelf')}` "
            f"env_error={fps.get('env_error')}"
        )
        lines.append("")
        lines.append("```")
        lines.append((fps.get("observation") or "").strip())
        lines.append("```")
        lines.append("")
        lines.append(f"action: `{fps.get('action')}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(fps.get("state_patch") or {}, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- n is 1–2 seeds on a toy 24-shelf warehouse, not SkillExecBench's 500.",
        "- Temperature 0.0 is still not bit-stable on this GPU for 14B "
        "(same seed has aborted at 4 steps or run longer).",
        "- History prompts start smaller because they do not serialize 24 shelves; "
        "they grow with t. SKILL.state stays ~flat.",
        "- `wall_s` includes prompt build. The first cell of each model is a cold "
        "Ollama load; skillstate seed 7 is always first, so history seed 7 is warm "
        "on that model. Seed 21 cells are warm if the model stayed resident.",
        "- Offline-scripted is a harness check, not a live Table 2 row.",
        "",
    ]
    return "\n".join(lines)
