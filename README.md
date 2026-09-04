# SKILL.state PoC

Local runtime for **[SKILL.state: Scalable Long-Horizon Agent Skills](https://arxiv.org/abs/2608.26263)** (Sanket Badhe, Priyanka Tiwari, Jonghyun Chung; arXiv:2608.26263, EMNLP). Talks to **Ollama** on a Framework Desktop (AMD Ryzen AI Max / Strix Halo, Linux or Windows).

This is an **independent proof of concept**. It is not the authors’ official code, not affiliated with Google or Purdue, and not a reproduction of SkillExecBench’s 500-shelf warehouse. The 24-shelf floor is sized so a 14B local model can emit valid JSON patches.

## What this is

Ordinary agent loops append every observation, thought, and action to the prompt. Prompt size grows with horizon \(T\); cumulative tokens grow as \(O(T^2)\). SKILL.state replaces that transcript with an explicit execution state.

At step \(t\) the model sees **only**

\[
A_t = (P, \Sigma_t, O_t)
\]

- \(P\) — immutable skill specification (markdown)
- \(\Sigma_t\) — current structured state (compact JSON)
- \(O_t\) — the latest environment observation

It writes throwaway chain-of-thought \(R_t\), then a JSON object with **exactly two keys**:

```json
{
  "state_patch": { "inventory": { "shelf_03": null } },
  "action": "SHIP item_12 shelf_03"
}
```

The runtime validates the patch against the skill schema **on a copy of \(\Sigma\)**, executes the action, and **commits \(\Sigma_{t+1} = \Sigma_t \oplus \Delta\Sigma_t\) only if the environment accepted the action** (deep merge; JSON `null` deletes that key). It then **discards \(R_t\)**. Next prompt does not contain prior observations, actions, or reasoning. Prompt size stays \(O(|P| + |\Sigma| + |O|)\).

The `compare` command runs this against an honest ReAct-history baseline on the **same warehouse seed**. SKILL.state’s prompt-token curve must stay roughly flat. History must climb. If it doesn’t, the implementation is wrong.

## Framework Desktop: run it

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and Ollama.

### 1. Install Ollama

Linux (including Fedora / Ubuntu on Framework Desktop):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows:

```powershell
winget install Ollama.Ollama
```

or [ollama.com/download/windows](https://ollama.com/download/windows).

Start the daemon if it is not already a service:

```bash
ollama serve
```

### 2. Pull a 14B-class model (not 70B)

SKILL.state lives or dies on reliable JSON patches. Start here:

```bash
ollama pull qwen2.5:14b
```

| Unified memory | Suggested models |
|---|---|
| 32GB | `qwen2.5:14b` or `llama3.1:8b` |
| 64GB | `qwen2.5:32b` or `qwen3:30b` (if available) |
| 128GB | `llama3.3:70b` or `qwen2.5:72b` |

A slow 70B that overwrites the wrong shelf just looks like a broken warehouse.

### 3. Install this repo

```bash
git clone https://github.com/PixnBits/skill-state-poc
cd skill-state-poc
uv sync
```

Copy `.env.example` to `.env` if you want to change the model or bind address. There are no API keys.

### 4. Warehouse demo (CLI)

```bash
uv run skillstate warehouse
```

Cap the horizon while you are checking the plumbing:

```bash
uv run skillstate warehouse --max-steps 12
```

Each turn prints:

```
step | prompt tokens | state JSON | observation | action | patch | running totals
```

**Success looks like:** prompt tokens stay in a narrow band (same order of magnitude from step 0 to step 12). Actions are `STORE` / `SHIP` / `WAIT` / `DONE`. The last line says `SUCCESS` once inbound is empty and every customer order is in `shipped`. Reasoning streams in a panel titled `R_t` and is **not** copied into the next prompt.

Offline (no Ollama; scripted policy — useful to see the loop):

```bash
uv run skillstate warehouse --offline --compact --max-steps 8
```

### 5. Dashboard

```bash
uv run skillstate ui
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

**Success looks like:**

- **Left** — live \(\Sigma_t\) JSON. Keys touched this step are highlighted in amber.
- **Centre** — the current observation \(O_t\) and the last action \(a_t\) only. No transcript.
- **Right** — discarded reasoning for **this step**, stamped `discard`, plus a sparkline of prompt tokens vs step that stays flat.
- Buttons: **Start warehouse**, **Start repoops**, **Pause**, **Reset**, **Change model**.

No Node, no React, no build step. FastAPI serves `src/skillstate/ui/static/index.html`.

### 6. Point at another model

```bash
OLLAMA_MODEL=llama3.1:8b uv run skillstate warehouse
# or
uv run skillstate warehouse --model qwen2.5:32b
```

Same flags work for `ui` and `compare`. Temperature is hard-coded to `0.0`. Completions are capped at 2048 tokens so a grammar-failing ReAct dump cannot fill the 32k context. No tool/function-calling.

### 7. Compare mode (the actual proof)

```bash
uv run skillstate compare --max-steps 30
# or without Ollama:
uv run skillstate compare --offline --max-steps 16
```

Prints:

```
runtime     | success | avg prompt tokens | max prompt tokens | total tokens
skillstate  |  ...    |  ~flat            |  ~flat            |  O(T)
history     |  ...    |  climbing         |  climbing         |  O(T²-ish)
```

Writes `runs/<timestamp>.json`. If SKILL.state is not flat or history is not climbing, the process exits the table with `Proof FAILED`.

Equivalent scripts: `uv run python scripts/demo_warehouse.py`, `uv run python scripts/demo_compare.py`.

## Tests

```bash
uv run pytest
```

No Ollama. Covers merge (nested overwrite, null-delete, unknown keys), the 24-shelf env, a 5-step fake-LLM episode, validator retry/abort, env-reject desync (Σ unchanged), silent-drift recovery_lag, messy JSON extraction, the flat-vs-growing prompt-curve proof, and the bench JSON schema (`wall_s >= 0` on both runtimes).

## Architecture

Algorithm 1 from the paper (`src/skillstate/runtime.py`), with a commit rule this PoC is strict about:

1. Receive \(O_t\)
2. Construct prompt \((P, \Sigma_t, O_t)\)
3. Generate \((R_t, \Delta\Sigma_t, a_t)\)
4. Validate \(\Delta\Sigma_t\) against the pydantic skill schema (`extra="forbid"`) on a **copy** of \(\Sigma\)
5. Validate action grammar
6. Execute \(a_t\)
7. Commit \(\Sigma_{t+1} \leftarrow \Sigma_t \oplus \Delta\Sigma_t\) **only if** `info["valid"]` is true
8. Drop \(R_t\)

\(\oplus\) is RFC 7396 JSON Merge Patch (`src/skillstate/merge.py`): nested objects merge key-wise; arrays/scalars replace; JSON `null` deletes the key.

**Documented schema exception:** a `null` under `inventory.shelf_NN` deletes that key in \(\oplus\), but an empty shelf is still a declared slot. `WarehouseState` fills missing `shelf_00..shelf_23` keys back in as `null`. Unknown shelf ids and unknown **top-level** keys still fail. This PoC does not use `extra="ignore"`.

## State commit rule

\(\Sigma\) is the model’s belief. The env is ground truth for physics. A grammar-valid action that the warehouse rejects (`STORE` onto an occupied shelf, `SHIP` from the wrong one) must **not** update \(\Sigma\). The next prompt is still \(A_t = (P, \Sigma_{\text{unchanged}}, O_{\text{env error}})\). Schema/JSON junk still never calls `env.step`.

Do not “fix” desync by dumping the env into the prompt.

## Drift recovery

Paper Exp 3, miniature. Off by default.

```bash
uv run skillstate warehouse --max-steps 40 --seed 7 --drift-at 5
uv run skillstate compare --max-steps 80 --seed 7 --drift-at 10
```

After step index \(N\) commits, the env secretly `MOVE`s one pending item to another empty shelf. The observation is a cycle-count / floor-scanner line, not `ALERT: Another worker moved…`. The agent must notice \(O\) contradicts \(\Sigma\) and patch `inventory`. The CLI prints `recovery_lag` (steps from the drift until \(\Sigma\) matches env ground truth for that item, or `null` if it never does).

History has no \(\Sigma\). On ReAct-history rows, `recovery_lag` is steps after drift until a **valid** action names `to_shelf` for the drifted item (`SHIP item to_shelf` or `MOVE item * to_shelf`), or the env has shipped it. The item already sits on `to_shelf` in the environment the instant the silent MOVE happens, so `loc == to_shelf` without an action is **not** recovery. History rows have no `patch_class`; the classifier is SKILL.state-only.

## Cross-size bench (SKILL.state vs history)

Question: can a smaller local model under SKILL.state match or beat a larger model under full-conversation ReAct history, on success / drift recovery / wall-clock / tokens?

```bash
uv run skillstate bench --seeds 7,21 --max-steps 40 --drift-at 10
# default models = Tier A: qwen2.5:14b, qwen3.8:27b, gemma4:26b, gemma4:31b, granite4.2:30b
```

Identical protocol for every cell: warehouse, temperature 0.0, `--max-steps 40`, `--drift-at 10`. Each cell is `(model, runtime ∈ {skillstate, history}, seed)`. Run order per model is skillstate seed 7, history seed 7, skillstate seed 21, history seed 21 (skillstate first so cold-load cost is not dumped only onto history). Does **not** stop after the first recovery (`sweep` still does; that stop condition is wrong for this comparison). `--drift-at` stays off on `warehouse` / `compare` unless you pass it.

Traces: `runs/bench/<model>__<runtime>__seed<N>.json`. Summary tables: [`runs/BENCH.md`](runs/BENCH.md). Offline-scripted is a harness check, not a live Table 2 row. These are this machine's numbers, not the paper's 16× token table.

## Warehouse skill

24 shelves (`shelf_00` … `shelf_23`). Seeded scenario: inbound shipments, customer orders, occasional “another worker moved the item” drift.

```
STORE <item> <shelf>
SHIP  <item> <shelf>
MOVE  <item> <from_shelf> <to_shelf>
WAIT
DONE
```

State shape:

```json
{
  "inventory": { "shelf_00": "item_12", "shelf_01": null },
  "inbound": [],
  "pending_orders": ["item_12"],
  "shipped": [],
  "last_action": "STORE item_03 shelf_02",
  "step": 4
}
```

`inbound` is the sufficient-statistic slot for shipments not stored on the same turn (otherwise \(O_t\) is discarded and the item is forgotten). Success = every order shipped, inbound empty, pending empty.

## Repo-ops skill

Toy git: one file per branch, PRs, CI. Grammar: `BRANCH`, `COMMIT <branch> add_logging|fix_ci`, `CREATE_PR`, `RUN_CI`, `MERGE`, `WAIT`, `DONE`.

## Ollama down?

The CLI health-checks `http://127.0.0.1:11434` on startup and prints the exact install + `ollama pull` commands, then exits 1. Nothing is sent off-box.

## License

MIT. Paper: [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) (CC BY 4.0). Cite the paper; don’t cite this repo as the official artifact.
