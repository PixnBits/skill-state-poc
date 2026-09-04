# Agent rules for this repository

This is an independent PoC of **SKILL.state** (Badhe, Tiwari, Chung; arXiv:2608.26263). It is not the authors’ code.

## Runtime contract (do not “simplify” this)

At every step `t` the model receives **only**

```
A_t = (P, Σ_t, O_t)
```

- `P` — immutable skill markdown (`skills/*/skill.md`)
- `Σ_t` — current structured state as compact JSON
- `O_t` — the latest environment observation **only**

The model emits throwaway reasoning `R_t` plus a JSON object with **exactly two keys**:

```json
{ "state_patch": { }, "action": "<cmd>" }
```

Then the runtime: parse → validate against the skill schema → `Σ_{t+1} = Σ_t ⊕ ΔΣ_t` (deep merge, JSON `null` deletes that key) → execute `action` → **discard `R_t`**. Never append prior observations, actions, or chain-of-thought to the next SKILL.state prompt.

Validator failure: do not apply the patch, do not execute the action, retry once with the error as `O_{t+1}`. Two consecutive failures end the episode as failed.

## Where the pieces live

| Concern | File |
|---|---|
| Algorithm 1 loop | `src/skillstate/runtime.py` |
| ⊕ merge | `src/skillstate/merge.py` |
| Prompt templates (paper Appendix A) | `src/skillstate/prompts.py` |
| Honest ReAct baseline | `src/skillstate/history_runtime.py` |
| Schema + extra=`forbid` | `src/skillstate/schemas.py` and `skills/*/schema.py` |

The `history` runtime **must** grow the prompt. If a change makes SKILL.state prompts climb with `t`, or history stay flat, the implementation is wrong.

## Tests

`uv run pytest` must pass **without Ollama**. Do not skip the prompt-curve assertion in `tests/test_runtime_offline.py`.

## Local LLM

Ollama on `http://127.0.0.1:11434/v1`. Default `OLLAMA_MODEL=qwen2.5:14b`. Temperature 0. No tools. No cloud APIs, no telemetry, no secrets.
