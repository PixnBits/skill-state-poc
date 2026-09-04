"""Skill registry. Each skill is (P, schema, env)."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from pydantic import BaseModel


class Env(Protocol):
    def reset(self) -> str: ...
    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]: ...
    def success(self) -> bool: ...
    def snapshot(self) -> dict[str, Any]: ...


class Skill(Protocol):
    name: str
    instructions: str
    state_schema: type[BaseModel]

    def initial_state(self) -> dict[str, Any]: ...
    def parse_action(self, command: str) -> Any | None: ...
    def make_env(self, seed: int, **kwargs: Any) -> Env: ...


def load_skill(name: str) -> Skill:
    key = name.strip().lower()
    if key in {"warehouse", "wh"}:
        from skillstate.skills.warehouse import WarehouseSkill

        return WarehouseSkill()
    if key in {"repoops", "repo", "git"}:
        from skillstate.skills.repoops import RepoOpsSkill

        return RepoOpsSkill()
    raise KeyError(f"unknown skill {name!r}; expected 'warehouse' or 'repoops'")


PolicyFn = Callable[[str], str]
