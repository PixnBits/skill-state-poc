"""Toy git: branches, one file, PRs, CI. Deterministic, no real git."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from skillstate.skills.repoops.schema import initial_repoops_state

Op = Literal[
    "BRANCH", "COMMIT", "CREATE_PR", "RUN_CI", "MERGE", "WAIT", "DONE"
]

LOG_SNIPPET = "log('start')\n"
FIX_MARKER = "# ci-ok\n"
MAIN_FILE = "print('hello')\n"


@dataclass(frozen=True)
class RepoAction:
    op: Op
    name: str | None = None
    payload: str | None = None
    pr_id: int | None = None
    raw: str = ""


_BRANCH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/-]{0,31}$")


def parse_repoops_action(command: str) -> RepoAction | None:
    if command is None:
        return None
    parts = command.strip().split()
    if not parts:
        return None
    op = parts[0].upper()
    if op == "WAIT" and len(parts) == 1:
        return RepoAction(op="WAIT", raw="WAIT")
    if op == "DONE" and len(parts) == 1:
        return RepoAction(op="DONE", raw="DONE")
    if op == "BRANCH" and len(parts) == 2 and _BRANCH_RE.match(parts[1]):
        return RepoAction(op="BRANCH", name=parts[1], raw=f"BRANCH {parts[1]}")
    if op == "COMMIT" and len(parts) == 3 and parts[2] in {"add_logging", "fix_ci"}:
        return RepoAction(
            op="COMMIT",
            name=parts[1],
            payload=parts[2],
            raw=f"COMMIT {parts[1]} {parts[2]}",
        )
    if op == "CREATE_PR" and len(parts) == 2:
        return RepoAction(op="CREATE_PR", name=parts[1], raw=f"CREATE_PR {parts[1]}")
    if op == "RUN_CI" and len(parts) == 2:
        return RepoAction(op="RUN_CI", name=parts[1], raw=f"RUN_CI {parts[1]}")
    if op == "MERGE" and len(parts) == 2 and parts[1].isdigit():
        return RepoAction(
            op="MERGE", pr_id=int(parts[1]), raw=f"MERGE {parts[1]}"
        )
    return None


@dataclass
class RepoOpsEnv:
    seed: int = 42
    horizon: int = 16
    t: int = 0
    branches: dict[str, dict[str, str]] = field(default_factory=dict)
    prs: list[dict[str, Any]] = field(default_factory=list)
    tickets: list[str] = field(default_factory=list)
    next_pr_id: int = 1
    ticket_fired: bool = False
    n_valid_actions: int = 0
    n_invalid_actions: int = 0

    def reset(self) -> str:
        init = initial_repoops_state()
        self.t = 0
        self.branches = {
            "main": {"file": MAIN_FILE, "ci": "pass"},
        }
        self.prs = []
        self.tickets = []
        self.next_pr_id = 1
        self.ticket_fired = False
        self.n_valid_actions = 0
        self.n_invalid_actions = 0
        self.ticket_fired = True
        self.tickets.append("OP-1")
        return (
            f"Shift start (seed {self.seed}, horizon {self.horizon}). "
            "Ticket OP-1: create a feature branch, COMMIT add_logging, "
            "CREATE_PR, RUN_CI (fix_ci + RUN_CI if it fails), MERGE, then DONE."
        )

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        parsed = parse_repoops_action(action)
        if parsed is None:
            self.n_invalid_actions += 1
            result = f"ERROR: action does not match repoops grammar: {action!r}"
            valid = False
        else:
            result, valid = self._apply(parsed)
            if valid:
                self.n_valid_actions += 1
            else:
                self.n_invalid_actions += 1
        self.t += 1
        done = self.t >= self.horizon
        if parsed is not None and parsed.op == "DONE" and self.success():
            done = True
        info = {"valid": valid, "success": self.success(), "gt": self.snapshot(), "t": self.t}
        return result, done, info

    def success(self) -> bool:
        merged = [p for p in self.prs if p["status"] == "merged"]
        if not merged:
            return False
        main = self.branches.get("main")
        if not main:
            return False
        return LOG_SNIPPET.strip() in main["file"] and main["ci"] == "pass"

    def snapshot(self) -> dict[str, Any]:
        return {
            "branches": {k: dict(v) for k, v in self.branches.items()},
            "prs": [dict(p) for p in self.prs],
            "tickets": list(self.tickets),
            "t": self.t,
        }

    def _apply(self, action: RepoAction) -> tuple[str, bool]:
        if action.op == "WAIT":
            return "OK: waited.", True
        if action.op == "DONE":
            if self.success():
                return "OK: feature merged to main.", True
            return f"ERROR: cannot DONE. repo={self.snapshot()}", False
        if action.op == "BRANCH":
            name = action.name or ""
            if name in self.branches:
                return f"ERROR: branch {name} already exists.", False
            src = self.branches["main"]
            self.branches[name] = {"file": src["file"], "ci": "unknown"}
            return f"Success: created branch {name} from main.", True
        if action.op == "COMMIT":
            name = action.name or ""
            if name not in self.branches:
                return f"ERROR: no branch {name}.", False
            if name == "main":
                return "ERROR: commit directly to main is forbidden.", False
            file_text = self.branches[name]["file"]
            if action.payload == "add_logging":
                if LOG_SNIPPET not in file_text:
                    file_text = file_text.rstrip("\n") + "\n" + LOG_SNIPPET
            elif action.payload == "fix_ci":
                if FIX_MARKER not in file_text:
                    file_text = file_text.rstrip("\n") + "\n" + FIX_MARKER
            else:
                return "ERROR: unknown commit payload.", False
            self.branches[name]["file"] = file_text
            self.branches[name]["ci"] = "pending"
            return f"Success: committed {action.payload} on {name}.", True
        if action.op == "CREATE_PR":
            name = action.name or ""
            if name not in self.branches or name == "main":
                return f"ERROR: cannot open PR from {name}.", False
            for pr in self.prs:
                if pr["source"] == name and pr["status"] == "open":
                    return f"ERROR: open PR already exists for {name}.", False
            pr = {
                "id": self.next_pr_id,
                "source": name,
                "target": "main",
                "status": "open",
            }
            self.next_pr_id += 1
            self.prs.append(pr)
            return f"Success: opened PR #{pr['id']} {name} → main.", True
        if action.op == "RUN_CI":
            name = action.name or ""
            if name not in self.branches:
                return f"ERROR: no branch {name}.", False
            file_text = self.branches[name]["file"]
            ok = LOG_SNIPPET.strip() in file_text and FIX_MARKER.strip() in file_text
            self.branches[name]["ci"] = "pass" if ok else "fail"
            if ok:
                return f"Success: CI passed on {name}.", True
            return (
                f"CI failed on {name}: missing log line or CI-fix marker. "
                "COMMIT fix_ci then RUN_CI again.",
                True,
            )
        if action.op == "MERGE":
            pr_id = action.pr_id
            pr = next((p for p in self.prs if p["id"] == pr_id), None)
            if pr is None:
                return f"ERROR: no PR #{pr_id}.", False
            if pr["status"] != "open":
                return f"ERROR: PR #{pr_id} is {pr['status']}.", False
            src = self.branches.get(pr["source"])
            if src is None:
                return "ERROR: source branch missing.", False
            if src["ci"] != "pass":
                return f"ERROR: CI is {src['ci']} on {pr['source']}; cannot merge.", False
            self.branches["main"]["file"] = src["file"]
            self.branches["main"]["ci"] = "pass"
            pr["status"] = "merged"
            if "OP-1" in self.tickets:
                self.tickets.remove("OP-1")
            return f"Success: merged PR #{pr_id} into main.", True
        return "ERROR: unknown op.", False
