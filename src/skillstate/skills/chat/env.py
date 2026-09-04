"""Chat env: the next observation is the next human line, not a world tick."""

from __future__ import annotations

import queue
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from skillstate.logging_util import estimate_tokens
from skillstate.skills.chat.schema import initial_chat_state

Op = Literal["SAY", "ASK", "DONE"]

_ACTION_RE = re.compile(r"^(SAY|ASK|DONE) (.+)$", re.DOTALL)
_STOP = object()

OnWait = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ChatAction:
    op: Op
    text: str
    raw: str


def parse_chat_action(command: str) -> ChatAction | None:
    if not command or not isinstance(command, str):
        return None
    stripped = command.strip()
    if stripped.upper() == "WAIT":
        return None
    match = _ACTION_RE.match(stripped)
    if not match:
        return None
    op = match.group(1).upper()
    text = match.group(2).strip()
    if not text:
        return None
    if op not in {"SAY", "ASK", "DONE"}:
        return None
    return ChatAction(op=op, text=text, raw=f"{op} {text}")  # type: ignore[arg-type]


def project_history_prompt(
    instructions: str,
    prior_lines: list[tuple[str, str]],
    observation: str,
) -> str:
    """What a ReAct-history prompt *would* look like. Never sent to the model."""
    if prior_lines:
        history = "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {text}"
            for role, text in prior_lines
        )
    else:
        history = "(none)"
    return (
        f"{instructions.strip()}\n\n"
        f"History:\n{history}\n\n"
        f"Latest Observation: {observation}"
    )


def estimate_history_tokens(
    instructions: str,
    prior_lines: list[tuple[str, str]],
    observation: str,
) -> int:
    return estimate_tokens(project_history_prompt(instructions, prior_lines, observation))


@dataclass
class ChatEnv:
    """Physics: the human. step() blocks until feed() supplies the next O."""

    seed: int = 0
    horizon: int = 128
    t: int = 0
    closed: bool = False
    last_reply: str = ""
    last_op: str = ""
    n_valid_actions: int = 0
    n_invalid_actions: int = 0
    awaiting_user: bool = False
    reset_observation: str | None = None
    on_wait: OnWait | None = None
    _inbox: queue.Queue[Any] = field(default_factory=queue.Queue)

    def feed(self, text: str | None) -> None:
        """Queue the next user utterance. ``None`` unblocks a waiting step (reset)."""
        self._inbox.put(_STOP if text is None else text)

    def reset(self) -> str:
        self.t = 0
        self.closed = False
        self.last_reply = ""
        self.last_op = ""
        self.n_valid_actions = 0
        self.n_invalid_actions = 0
        self.awaiting_user = False
        # Human starts: O_0 is the first user line. run_skill_state calls
        # reset() before the first model call, so we block here until feed().
        # Do not drain _inbox — tests preload lines before the loop starts.
        if self.reset_observation:
            return self.reset_observation
        if self.on_wait:
            self.on_wait({"type": "awaiting_user", "assistant_text": None, "op": None})
        return self._wait_user()

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        parsed = parse_chat_action(action)
        if parsed is None:
            self.n_invalid_actions += 1
            self.t += 1
            return (
                f"ERROR: action does not match chat grammar: {action!r}. "
                "Emit SAY/ASK/DONE followed by the text the user should see.",
                False,
                {"valid": False, "assistant_text": None},
            )
        self.n_valid_actions += 1
        self.t += 1
        self.last_reply = parsed.text
        self.last_op = parsed.op
        info: dict[str, Any] = {
            "valid": True,
            "assistant_text": parsed.text,
            "op": parsed.op,
        }
        if parsed.op == "DONE":
            self.closed = True
            return f"Conversation ended: {parsed.text}", True, info
        self.awaiting_user = True
        if self.on_wait:
            self.on_wait({"type": "awaiting_user", "assistant_text": parsed.text, "op": parsed.op})
        next_obs = self._wait_user()
        return next_obs, False, info

    def _wait_user(self) -> str:
        self.awaiting_user = True
        try:
            item = self._inbox.get()
        finally:
            self.awaiting_user = False
        if item is _STOP:
            raise InterruptedError("reset")
        return str(item)

    def success(self) -> bool:
        return self.closed

    def snapshot(self) -> dict[str, Any]:
        snap = initial_chat_state()
        snap.update(
            {
                "t": self.t,
                "closed": self.closed,
                "last_reply": self.last_reply,
                "last_op": self.last_op,
                "n_valid_actions": self.n_valid_actions,
                "n_invalid_actions": self.n_invalid_actions,
                "awaiting_user": self.awaiting_user,
            }
        )
        return snap
