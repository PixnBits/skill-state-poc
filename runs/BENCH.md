# SKILL.state vs history bench

> **PARTIAL** — matrix stopped before every planned cell finished. Numbers below are from completed cells only.

Hardware: Framework Desktop, Ollama local, temperature 0.0. Date 2026-09-04T06:51:46.341436+00:00. Commit `4b10e90`.

Protocol: warehouse `--max-steps 40 --drift-at 10` seeds `[7, 21]`. Per model: skillstate then history on seed 7, then the same on seed 21 (first cell of a model is cold-load; later cells on that model are warm).

History has no Σ. `recovery_lag` on history rows is steps after drift until a *valid* action names `to_shelf` for the drifted item, or the env has shipped it. We do not invent a history `patch_class`. `loc == to_shelf` is already true immediately after the silent MOVE, so it does not count as recovery without an action.

These are this machine's numbers. Not the paper's 16× token table.

## Table 1 — full matrix

| model | params-ish | runtime | seed | success | recovered | recovery_lag | steps | wall_s | avg prompt | max prompt | total tokens | cold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b | 14B | skillstate | 7 | no | no | null | 4 | 62.22 | 1033.8 | 1089 | 4988 | yes |
| qwen2.5:14b | 14B | history | 7 | no | no | null | 11 | 114.434 | 1508.0 | 2245 | 17879 | no |
| qwen2.5:14b | 14B | skillstate | 21 | no | no | null | 5 | 82.471 | 1030.2 | 1090 | 6291 | no |
| qwen2.5:14b | 14B | skillstate | mean | 0.0 | 0.0 | null | 4.5 | 72.3 | 1032.0 | 1089.50 | 5640 | — |

## Table 2 — cross-size headline pairs

| smaller skillstate | larger history | seeds | success | recovery | wall_s ratio | token ratio | ss wins? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b skillstate | qwen3.8:27b history | — | *not run* | — | — | — | — |
| qwen3.8:27b skillstate | gemma4:31b history | — | *not run* | — | — | — | — |
| gemma4:26b skillstate | gemma4:31b history | — | *not run* | — | — | — | — |
| qwen3.8:27b skillstate | qwen3.6:35b history | — | *not run* | — | — | — | — |

A smaller skillstate model “wins” a pair if success and recovery are ≥ the larger history model and wall_s **or** total tokens are lower.

## Table 3 — skillstate failure taxonomy

History rows have no `patch_class`. Quotes below are skillstate only.

### qwen2.5:14b seed=7

reached_drift=False recovery_lag=None grammar_fail_count=2 delete_only=0 stale_location=0 correct_relocation=0 starved_correct=0 fail_reason="two consecutive validator failures: action does not match skill grammar: 'WAIT item_06'"

No cycle-count observation (grammar abort before drift, or no drift).

## Caveats

- n is 1–2 seeds on a toy 24-shelf warehouse, not SkillExecBench's 500.
- Temperature 0.0 is still not bit-stable on this GPU for 14B (same seed has aborted at 4 steps or run longer).
- History prompts start smaller because they do not serialize 24 shelves; they grow with t. SKILL.state stays ~flat.
- `wall_s` includes prompt build. The first cell of each model is a cold Ollama load; skillstate seed 7 is always first, so history seed 7 is warm on that model. Seed 21 cells are warm if the model stayed resident.
- Offline-scripted is a harness check, not a live Table 2 row.
