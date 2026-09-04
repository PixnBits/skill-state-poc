"""Exact SKILL.state and ReAct prompt templates (paper Appendix A).

SKILL.state prompt contains ONLY A_t = (P, Σ_t, O_t).
The history runtime appends every observation, thought, and action — that
growth is the baseline this PoC exists to measure.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Paper Appendix A.4, reconstructed as a single user message. Compact JSON
# (separators=(',', ':')) is intentional: it is how the paper serializes Σ.
SKILL_STATE_TEMPLATE = """Instructions:
{instructions}

Skill Execution State:
```json
{state_json}
```
Latest Observation: {observation}

Provide your response with:
1. Step-by-step reasoning (will be discarded after execution)
2. A JSON block fenced with json containing both your State Patch and your Action. The JSON block MUST have exactly these two keys: {{ "state_patch": {{ <dict: your state updates, set keys to null to delete> }}, "action": "<string: the exact command you want to execute>" }}
"""

# Paper Appendix A.1 Prompt Runtime (ReAct-style).
REACT_TEMPLATE = """Instructions:
{instructions}

History:
{history}

Latest Observation: {observation}

Generate your next reasoning and action (format 'Action: <cmd>'):
"""


def dump_state(state: dict[str, Any]) -> str:
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


def build_skill_state_prompt(
    instructions: str,
    state: dict[str, Any],
    observation: str,
) -> str:
    """Construct A_t = (P, Σ_t, O_t). Callers must not pass history."""
    return SKILL_STATE_TEMPLATE.format(
        instructions=instructions.strip(),
        state_json=dump_state(state),
        observation=observation,
    )


def build_react_prompt(
    instructions: str,
    history: list[str],
    observation: str,
) -> str:
    history_block = "\n\n".join(history) if history else "(none)"
    return REACT_TEMPLATE.format(
        instructions=instructions.strip(),
        history=history_block,
        observation=observation,
    )


def format_history_turn(observation: str, response: str) -> str:
    return f"Observation: {observation}\nReasoning & Action: {response}"


_STATE_BLOCK = re.compile(
    r"Skill Execution State:\s*```json\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)
_LATEST_OBS = re.compile(
    r"Latest Observation:\s*(.*?)(?:\n\nProvide your response|\n\nGenerate your next|\Z)",
    re.DOTALL,
)


def parse_skill_state_prompt(prompt: str) -> tuple[dict[str, Any], str]:
    """Recover (Σ_t, O_t) from a SKILL.state prompt. Used by the offline policy."""
    state_match = _STATE_BLOCK.search(prompt)
    if not state_match:
        raise ValueError("prompt is missing Skill Execution State block")
    state = json.loads(state_match.group(1))
    obs_match = _LATEST_OBS.search(prompt)
    if not obs_match:
        raise ValueError("prompt is missing Latest Observation")
    observation = obs_match.group(1).strip()
    return state, observation
