from __future__ import annotations

from pathlib import Path
from typing import Any

from skillstate.skills.warehouse.env import WarehouseEnv, parse_warehouse_action
from skillstate.skills.warehouse.schema import WarehouseState, initial_warehouse_state

_SKILL_MD = Path(__file__).with_name("skill.md")


class WarehouseSkill:
    name = "warehouse"
    state_schema = WarehouseState

    def __init__(self) -> None:
        self.instructions = _SKILL_MD.read_text(encoding="utf-8")

    def initial_state(self) -> dict[str, Any]:
        return initial_warehouse_state()

    def parse_action(self, command: str) -> Any | None:
        return parse_warehouse_action(command)

    def make_env(self, seed: int, **kwargs: Any) -> WarehouseEnv:
        return WarehouseEnv(seed=seed, **kwargs)
