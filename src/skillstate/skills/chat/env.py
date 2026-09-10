"""Chat env: the next observation is the next human line, not a world tick.

Tools are skill *actions* (one command per step), not Ollama function-calling.
TIME / CALC / HASH run in the env and return their result as the next O_t
so the model can SAY it. The human is not blocked for those turns.
"""

from __future__ import annotations

import ast
import hashlib
import operator
import queue
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from skillstate.logging_util import estimate_tokens
from skillstate.skills.chat.schema import initial_chat_state

Op = Literal["SAY", "ASK", "DONE", "TIME", "CALC", "HASH"]
TALK_OPS = frozenset({"SAY", "ASK", "DONE"})
TOOL_OPS = frozenset({"TIME", "CALC", "HASH"})

_STOP = object()
_CALC_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

OnWait = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ChatAction:
    op: Op
    text: str
    raw: str
    is_tool: bool = False


def parse_chat_action(command: str) -> ChatAction | None:
    if not command or not isinstance(command, str):
        return None
    stripped = command.strip()
    if not stripped or stripped.upper() == "WAIT":
        return None
    op, _, rest = stripped.partition(" ")
    op = op.upper()
    text = rest.strip()
    if op == "TIME" and not text:
        return ChatAction(op="TIME", text="", raw="TIME", is_tool=True)
    if op == "CALC" and text:
        return ChatAction(op="CALC", text=text, raw=f"CALC {text}", is_tool=True)
    if op == "HASH" and text:
        return ChatAction(op="HASH", text=text, raw=f"HASH {text}", is_tool=True)
    if op in TALK_OPS and text:
        return ChatAction(op=op, text=text, raw=f"{op} {text}", is_tool=False)  # type: ignore[arg-type]
    return None


def _eval_calc(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_calc(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BIN:
        left = _eval_calc(node.left)
        right = _eval_calc(node.right)
        return _CALC_BIN[type(node.op)](left, right)
    if isinstance(node, ast.Expression):
        return _eval_calc(node.body)
    raise ValueError("only + - * / // % and numbers are allowed")


def run_chat_tool(parsed: ChatAction) -> tuple[bool, str]:
    """Return (ok, result_text). Never raises."""
    if parsed.op == "TIME":
        return True, datetime.now().astimezone().isoformat(timespec="seconds")
    if parsed.op == "HASH":
        digest = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        return True, digest
    if parsed.op == "CALC":
        if len(parsed.text) > 120:
            return False, "expression too long"
        try:
            tree = ast.parse(parsed.text, mode="eval")
            value = _eval_calc(tree)
        except Exception as exc:  # noqa: BLE001 — surface as a tool error observation
            return False, str(exc)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return True, str(value)
    return False, f"unknown tool {parsed.op}"


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
    last_tool: dict[str, Any] | None = None
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
        self.last_tool = None
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
                "Emit SAY/ASK/DONE <text>, or TIME, CALC <expr>, HASH <text>.",
                False,
                {"valid": False, "assistant_text": None, "tool": None},
            )
        self.n_valid_actions += 1
        self.t += 1
        self.last_reply = parsed.text
        self.last_op = parsed.op
        if parsed.is_tool:
            ok, result = run_chat_tool(parsed)
            tool = {
                "name": parsed.op,
                "args": parsed.text,
                "ok": ok,
                "result": result,
                "raw": parsed.raw,
            }
            self.last_tool = tool
            if ok and parsed.op == "CALC":
                obs = f"TOOL CALC: {parsed.text} = {result}"
            elif ok:
                obs = f"TOOL {parsed.op}: {result}"
            else:
                obs = f"TOOL ERROR {parsed.op}: {result}"
            if self.on_wait:
                self.on_wait({"type": "tool", "tool": tool, "action": parsed.raw})
            return obs, False, {"valid": True, "assistant_text": None, "op": parsed.op, "tool": tool}
        self.last_tool = None
        info: dict[str, Any] = {
            "valid": True,
            "assistant_text": parsed.text,
            "op": parsed.op,
            "tool": None,
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
                "last_tool": dict(self.last_tool) if self.last_tool else None,
            }
        )
        return snap
