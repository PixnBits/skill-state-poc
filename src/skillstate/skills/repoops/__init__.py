from __future__ import annotations

from pathlib import Path
from typing import Any

from skillstate.skills.repoops.env import RepoOpsEnv, parse_repoops_action
from skillstate.skills.repoops.schema import RepoOpsState, initial_repoops_state

_SKILL_MD = Path(__file__).with_name("skill.md")


class RepoOpsSkill:
    name = "repoops"
    state_schema = RepoOpsState

    def __init__(self) -> None:
        self.instructions = _SKILL_MD.read_text(encoding="utf-8")

    def initial_state(self) -> dict[str, Any]:
        return initial_repoops_state()

    def parse_action(self, command: str) -> Any | None:
        return parse_repoops_action(command)

    def make_env(self, seed: int, **kwargs: Any) -> RepoOpsEnv:
        kwargs.pop("compact", None)
        kwargs.pop("n_shelves", None)
        return RepoOpsEnv(seed=seed, **{k: v for k, v in kwargs.items() if k in {"horizon"}})
