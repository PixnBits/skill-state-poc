from __future__ import annotations

from pathlib import Path
from typing import Any

from skillstate.skills.chat.env import ChatEnv, parse_chat_action
from skillstate.skills.chat.schema import ChatState, initial_chat_state

_SKILL_MD = Path(__file__).with_name("skill.md")


class ChatSkill:
    name = "chat"
    state_schema = ChatState

    def __init__(self) -> None:
        self.instructions = _SKILL_MD.read_text(encoding="utf-8")

    def initial_state(self) -> dict[str, Any]:
        return initial_chat_state()

    def parse_action(self, command: str) -> Any | None:
        return parse_chat_action(command)

    def make_env(self, seed: int = 0, **kwargs: Any) -> ChatEnv:
        env = ChatEnv(seed=seed, horizon=int(kwargs.get("horizon") or 128))
        if kwargs.get("reset_observation"):
            env.reset_observation = str(kwargs["reset_observation"])
        return env
