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

Then the runtime: parse → validate the patch against the skill schema **on a copy of Σ** → validate action grammar → `env.step(action)` → **commit `Σ ← Σ ⊕ ΔΣ` only if `info["valid"]` is true** → **discard `R_t`**. Never append prior observations, actions, or chain-of-thought to the next SKILL.state prompt. Never commit Σ until `env.step` is valid. Never copy the env world into the prompt; Σ is the model’s belief, the env is physics.

Schema/JSON failure: do not merge, do not execute, retry once with the error as `O_{t+1}`. Two consecutive schema failures end the episode as failed. An env-rejected action is not a schema failure: leave Σ unchanged and pass the env error as the next observation.

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
