# Model sweep notes

Generated 2026-09-04. Protocol: warehouse `seed=7` `max-steps=40` `drift-at=10`, temperature **0.0**, SKILL.state runtime (no second loop).

P (`src/skillstate/skills/warehouse/skill.md`) already says cycle-count / scanner observations override Σ (`If O contradicts Σ, patch inventory to match O`, plus the Cycle count bullet). **No extra “trust the scanner” sentence was added.**

Commit rule unchanged: proposed Σ is classified even when `env.step` rejects; live Σ is still committed only on a valid action. `merge.py`, `extra="forbid"`, and the SKILL.state prompt template were not modified.

`--drift-at` remains off by default.

`gemma4:31b` and `granite4.2:30b` were **not** started: `qwen3.8:27b` recovered. `muse-glimmer` was skipped. `qwen3.6:35b` / `qwen3.8:27b-q8_0` were not started (a 27B finished an episode).

## Table

| model | reached_drift | steps | fail_reason | recovery_lag | grammar_fail_count | delete_only | stale_location | correct_relocation | starved_correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline-scripted | yes | 23 | null | 1 | 0 | 0 | 0 | 1 | 0 |
| qwen3.8:27b | yes | 20 | null | 1 | 0 | 0 | 0 | 1 | 0 |

Prompt curves stayed flat (skillstate O(1)):

- offline: `[1030, 1007, …, 1025]` max 1040
- qwen3.8:27b: `[1030, 1007, …, 1025]` max 1038

## First post-scanner action (quoted from traces)

### offline-scripted

reached_drift=yes  recovery_lag=1  fail_reason=None  prompt_flat=yes

step 11, patch_class=`correct_relocation`, env_error=False

Observation:

```
Success: Stored item_05 on shelf_03.
Customer ordered item_00.
Cycle count: item_00 is not on shelf_00. Floor scanner: item_00 is on shelf_05.
```

action: `WAIT`

state_patch:

```json
{
  "step": 12,
  "inventory": {
    "shelf_00": null,
    "shelf_05": "item_00"
  },
  "inbound": [],
  "pending_orders": ["item_06", "item_07", "item_02", "item_04", "item_00"],
  "shipped": [],
  "last_action": "WAIT"
}
```

Classification: **correct_relocation** (item written onto `to_shelf`, cleared from `from_shelf`; WAIT so the commit rule applied the patch).

### qwen3.8:27b

reached_drift=yes  recovery_lag=1  fail_reason=None  prompt_flat=yes

Drift: `item_00` `shelf_00` → `shelf_02` at step 10.

step 11, patch_class=`correct_relocation`, proposed_item_shelf=null (emptied by the SHIP), env_error=False

Observation:

```
Success: Stored item_05 on shelf_05.
Customer ordered item_00.
Cycle count: item_00 is not on shelf_00. Floor scanner: item_00 is on shelf_02.
```

action: `SHIP item_00 shelf_02`

state_patch:

```json
{
  "inventory": {
    "shelf_00": null,
    "shelf_02": null
  },
  "pending_orders": ["item_06", "item_07"],
  "shipped": ["item_02", "item_04", "item_00"],
  "last_action": "SHIP item_00 shelf_02",
  "step": 12
}
```

Classification: **correct_relocation** (SHIP targeted the scanner shelf `shelf_02`, not the stale `shelf_00`). Env accepted; `starved_correct=0`. Contrast with qwen2.5:14b, which on the same protocol emitted `{inventory:{shelf_00:null}}` + `SHIP item_00 shelf_00` (`delete_only`).

## Commands

```bash
uv run skillstate warehouse --offline --seed 7 --max-steps 40 --drift-at 10
uv run skillstate warehouse --model qwen3.8:27b --seed 7 --max-steps 40 --drift-at 10
uv run skillstate sweep   # same protocol; stops after first recovery
```
