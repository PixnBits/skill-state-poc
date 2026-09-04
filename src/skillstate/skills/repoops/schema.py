"""Repo-ops execution-state schema (paper Env 2, toy scale)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BranchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    ci: Literal["unknown", "pending", "pass", "fail"] = "unknown"


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    source: str
    target: str = "main"
    status: Literal["open", "merged", "closed"] = "open"


class RepoOpsState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branches: dict[str, BranchState]
    prs: list[PullRequest] = Field(default_factory=list)
    tickets: list[str] = Field(default_factory=list)
    last_action: str = ""
    step: int = 0


def initial_repoops_state() -> dict[str, Any]:
    return RepoOpsState(
        branches={"main": BranchState(file="print('hello')\n", ci="pass")},
        prs=[],
        tickets=[],
        last_action="",
        step=0,
    ).model_dump(mode="json")
