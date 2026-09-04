# SKILL.state vs history bench

Hardware: Framework Desktop, Ollama local, temperature 0.0. Date 2026-09-04T10:27:22.921576+00:00. Commit `97c1651`.

Protocol: warehouse `--max-steps 40 --drift-at 10` seeds `[7, 21]`. Per model: skillstate then history on seed 7, then the same on seed 21 (first cell of a model is cold-load; later cells on that model are warm).

History has no Σ. `recovery_lag` on history rows is steps after drift until a *valid* action names `to_shelf` for the drifted item, or the env has shipped it. We do not invent a history `patch_class`. `loc == to_shelf` is already true immediately after the silent MOVE, so it does not count as recovery without an action.

These are this machine's numbers. Not the paper's 16× token table.

**Answer (Tier A, this machine):** a smaller SKILL.state model *can* match a larger history model on success and recovery and beat it on tokens — `qwen3.8:27b` skillstate vs `gemma4:31b` history, both seeds, token ratio 0.40. It is not automatic: `qwen2.5:14b` skillstate did not match `qwen3.8:27b` history (14B grammar-aborted both seeds before drift), and `gemma4:26b` skillstate did not match `gemma4:31b` history. Wall-clock did not favor SKILL.state here (skillstate seed 7 is always the cold load; ~20-step history has not grown enough to dominate). Tier B / 35B was not started.

## Table 1 — full matrix

| model | params-ish | runtime | seed | success | recovered | recovery_lag | steps | wall_s | avg prompt | max prompt | total tokens | cold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b | 14B | skillstate | 7 | no | no | null | 4 | 62.22 | 1033.8 | 1089 | 4988 | yes |
| qwen2.5:14b | 14B | history | 7 | no | no | null | 11 | 114.434 | 1508.0 | 2245 | 17879 | no |
| qwen2.5:14b | 14B | skillstate | 21 | no | no | null | 5 | 82.471 | 1030.2 | 1090 | 6291 | no |
| qwen2.5:14b | 14B | history | 21 | yes | yes | 7 | 22 | 105.02 | 1744.1 | 2933 | 39901 | no |
| qwen3.8:27b | 27B | skillstate | 7 | yes | yes | 1 | 20 | 976.75 | 1018.2 | 1038 | 24061 | yes |
| qwen3.8:27b | 27B | history | 7 | yes | yes | 1 | 20 | 631.515 | 2268.3 | 3842 | 47961 | no |
| qwen3.8:27b | 27B | skillstate | 21 | yes | yes | 1 | 18 | 862.52 | 1017.0 | 1039 | 21870 | no |
| qwen3.8:27b | 27B | history | 21 | yes | yes | 6 | 18 | 539.007 | 2086.8 | 3422 | 39787 | no |
| gemma4:26b | 26B | skillstate | 7 | no | no | null | 6 | 313.298 | 1032.3 | 1081 | 6731 | yes |
| gemma4:26b | 26B | history | 7 | no | no | null | 2 | 122.981 | 889.0 | 944 | 1856 | no |
| gemma4:26b | 26B | skillstate | 21 | no | no | null | 7 | 262.037 | 1020.7 | 1081 | 7667 | no |
| gemma4:26b | 26B | history | 21 | no | no | null | 7 | 216.714 | 1044.3 | 1219 | 7544 | no |
| gemma4:31b | 31B | skillstate | 7 | no | no | null | 7 | 568.999 | 1037.9 | 1088 | 10613 | yes |
| gemma4:31b | 31B | history | 7 | yes | yes | 1 | 20 | 738.489 | 2798.8 | 4996 | 59860 | no |
| gemma4:31b | 31B | skillstate | 21 | yes | yes | 2 | 21 | 1511.05 | 1027.5 | 1136 | 29638 | no |
| gemma4:31b | 31B | history | 21 | yes | yes | 8 | 20 | 665.424 | 2553.3 | 4570 | 54450 | no |
| granite4.2:30b | 30B | skillstate | 7 | no | no | null | 2 | 473.434 | 1066.0 | 1102 | 2132 | yes |
| granite4.2:30b | 30B | history | 7 | no | no | null | 2 | 456.659 | 848.0 | 862 | 1696 | no |
| granite4.2:30b | 30B | skillstate | 21 | no | no | null | 2 | 458.954 | 1066.5 | 1102 | 2133 | no |
| granite4.2:30b | 30B | history | 21 | no | no | null | 2 | 457.126 | 848.5 | 863 | 1697 | no |
| qwen2.5:14b | 14B | skillstate | mean | 0.0 | 0.0 | null | 4.5 | 72.3 | 1032.0 | 1089.50 | 5640 | — |
| qwen2.5:14b | 14B | history | mean | 0.5 | 0.5 | 7.00 | 16.5 | 109.7 | 1626.0 | 2589 | 28890 | — |
| qwen3.8:27b | 27B | skillstate | mean | 1.0 | 1.0 | 1.00 | 19.0 | 919.6 | 1017.6 | 1038.50 | 22966 | — |
| qwen3.8:27b | 27B | history | mean | 1.0 | 1.0 | 3.50 | 19.0 | 585.3 | 2177.6 | 3632 | 43874 | — |
| gemma4:26b | 26B | skillstate | mean | 0.0 | 0.0 | null | 6.5 | 287.7 | 1026.5 | 1081 | 7199 | — |
| gemma4:26b | 26B | history | mean | 0.0 | 0.0 | null | 4.5 | 169.8 | 966.6 | 1081.50 | 4700 | — |
| gemma4:31b | 31B | skillstate | mean | 0.5 | 0.5 | 2.00 | 14.0 | 1040.0 | 1032.7 | 1112 | 20126 | — |
| gemma4:31b | 31B | history | mean | 1.0 | 1.0 | 4.50 | 20.0 | 702.0 | 2676.1 | 4783 | 57155 | — |
| granite4.2:30b | 30B | skillstate | mean | 0.0 | 0.0 | null | 2.0 | 466.2 | 1066.2 | 1102 | 2132 | — |
| granite4.2:30b | 30B | history | mean | 0.0 | 0.0 | null | 2.0 | 456.9 | 848.2 | 862.50 | 1696 | — |

## Table 2 — cross-size headline pairs

| smaller skillstate | larger history | seeds | success | recovery | wall_s ratio | token ratio | ss wins? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b skillstate | qwen3.8:27b history | 7 | ss=False h=True match=False | ss=False h=True match=False | 0.099 | 0.104 | no |
| qwen2.5:14b skillstate | qwen3.8:27b history | 21 | ss=False h=True match=False | ss=False h=True match=False | 0.153 | 0.158 | no |
| qwen3.8:27b skillstate | gemma4:31b history | 7 | ss=True h=True match=True | ss=True h=True match=True | 1.323 | 0.402 | yes |
| qwen3.8:27b skillstate | gemma4:31b history | 21 | ss=True h=True match=True | ss=True h=True match=True | 1.296 | 0.402 | yes |
| gemma4:26b skillstate | gemma4:31b history | 7 | ss=False h=True match=False | ss=False h=True match=False | 0.424 | 0.112 | no |
| gemma4:26b skillstate | gemma4:31b history | 21 | ss=False h=True match=False | ss=False h=True match=False | 0.394 | 0.141 | no |
| qwen3.8:27b skillstate | qwen3.6:35b history | — | *not run* | — | — | — | — |

A smaller skillstate model “wins” a pair if success and recovery are ≥ the larger history model and wall_s **or** total tokens are lower.

## Table 3 — skillstate failure taxonomy

History rows have no `patch_class`. Quotes below are skillstate only.

### qwen2.5:14b seed=7

reached_drift=False recovery_lag=None grammar_fail_count=2 delete_only=0 stale_location=0 correct_relocation=0 starved_correct=0 fail_reason="two consecutive validator failures: action does not match skill grammar: 'WAIT item_06'"

No cycle-count observation (grammar abort before drift, or no drift).

### qwen3.8:27b seed=7

reached_drift=True recovery_lag=1 grammar_fail_count=0 delete_only=0 stale_location=0 correct_relocation=1 starved_correct=0 fail_reason=None

step 11 patch_class=`correct_relocation` proposed_item_shelf=`None` env_error=False

```
Success: Stored item_05 on shelf_05.
Customer ordered item_00.
Cycle count: item_00 is not on shelf_00. Floor scanner: item_00 is on shelf_02.
```

action: `SHIP item_00 shelf_02`

```json
{
  "inventory": {
    "shelf_00": null,
    "shelf_02": null
  },
  "pending_orders": [
    "item_06",
    "item_07"
  ],
  "shipped": [
    "item_02",
    "item_04",
    "item_00"
  ],
  "last_action": "SHIP item_00 shelf_02",
  "step": 12
}
```

### gemma4:26b seed=7

reached_drift=False recovery_lag=None grammar_fail_count=3 delete_only=0 stale_location=0 correct_relocation=0 starved_correct=0 fail_reason="two consecutive validator failures: JSON must have exactly two keys: state_patch and action; got ['shelf_02']"

No cycle-count observation (grammar abort before drift, or no drift).

### gemma4:31b seed=7

reached_drift=False recovery_lag=None grammar_fail_count=3 delete_only=0 stale_location=0 correct_relocation=0 starved_correct=0 fail_reason="two consecutive validator failures: action does not match skill grammar: 'ORDER item_07 shelf_00'"

No cycle-count observation (grammar abort before drift, or no drift).

### granite4.2:30b seed=7

reached_drift=False recovery_lag=None grammar_fail_count=2 delete_only=0 stale_location=0 correct_relocation=0 starved_correct=0 fail_reason='two consecutive validator failures: empty model output'

No cycle-count observation (grammar abort before drift, or no drift).

## Caveats

- n is 1–2 seeds on a toy 24-shelf warehouse, not SkillExecBench's 500.
- Temperature 0.0 is still not bit-stable on this GPU for 14B (same seed has aborted at 4 steps or run longer).
- History prompts start smaller because they do not serialize 24 shelves; they grow with t. SKILL.state stays ~flat.
- `wall_s` includes prompt build. The first cell of each model is a cold Ollama load; skillstate seed 7 is always first, so history seed 7 is warm on that model. Seed 21 cells are warm if the model stayed resident.
- Offline-scripted is a harness check, not a live Table 2 row.
- Completions are capped at 2048 tokens (`max_tokens` on the Ollama OpenAI-compat call). That is not a prompt-template change; it stops a runaway ReAct dump from filling the 32k context. The first 14B history seed-21 attempt generated 30k+ tokens at ~9 tok/s before this cap; that cell was restarted under 2048.
- Tier B (`gemma4:e4b`, `qwen3-coder:30b`, `nemotron3:33b`, `qwen3.6:35b`) was not started. Tier A (14/27/26/31/Granite) finished 20/20 cells; that is the comparison this run was for. `qwen3.8:27b` default quant was not suspiciously fast or slow versus 31B, so `qwen3.8:27b-q8_0` was skipped. A Quantum ESPRESSO `ph.x` job was still on the box.
