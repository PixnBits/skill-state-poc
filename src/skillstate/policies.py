"""Scripted policies that emit valid SKILL.state / ReAct payloads.

These are the offline 'model'. They reconstruct belief from (Σ, O) for
SKILL.state, and from the full prompt transcript for the history baseline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from skillstate.prompts import parse_skill_state_prompt
from skillstate.skills.warehouse.env import parse_warehouse_action
from skillstate.skills.warehouse.schema import SHELF_IDS, empty_inventory

_SHIPMENT = re.compile(r"Shipment arrived containing (item_\d{2})")
_ORDER = re.compile(r"Customer ordered (item_\d{2})")
_DRIFT = re.compile(
    r"Another worker moved (item_\d{2}) from (shelf_\d{2}) to (shelf_\d{2})"
)
_CYCLE = re.compile(
    r"Cycle count: (item_\d{2}) is not on (shelf_\d{2})\. "
    r"Floor scanner: (item_\d{2}) is on (shelf_\d{2})\."
)


def _first_empty(inventory: dict[str, Any]) -> str | None:
    for shelf in SHELF_IDS:
        if inventory.get(shelf) is None:
            return shelf
    return None


def _find_item(inventory: dict[str, Any], item: str) -> str | None:
    for shelf, value in inventory.items():
        if value == item:
            return shelf
    return None


def warehouse_skillstate_policy(prompt: str) -> str:
    state, observation = parse_skill_state_prompt(prompt)
    inventory = dict(state.get("inventory") or empty_inventory())
    inbound = list(state.get("inbound") or [])
    pending = list(state.get("pending_orders") or [])
    shipped = list(state.get("shipped") or [])
    step = int(state.get("step") or 0) + 1
    patch: dict[str, Any] = {"step": step}
    inv_patch: dict[str, Any] = {}

    for item in _SHIPMENT.findall(observation):
        if item not in inbound and item not in inventory.values() and item not in shipped:
            inbound.append(item)
    for item in _ORDER.findall(observation):
        if item not in pending and item not in shipped:
            pending.append(item)
    for item, src, dst in _DRIFT.findall(observation):
        if inventory.get(src) == item or inventory.get(src) is not None:
            inv_patch[src] = None
            inventory[src] = None
        inv_patch[dst] = item
        inventory[dst] = item
    for item, src, scanned_item, dst in _CYCLE.findall(observation):
        if scanned_item != item:
            continue
        inv_patch[src] = None
        inventory[src] = None
        inv_patch[dst] = item
        inventory[dst] = item

    action = "WAIT"
    reasoning = "Scripted policy: apply observation, then STORE / SHIP / DONE."

    if inbound:
        item = inbound[0]
        shelf = _first_empty(inventory)
        if shelf is None:
            action = "WAIT"
            reasoning = "Dock is occupied and no empty shelf; WAIT."
        else:
            inventory[shelf] = item
            inbound = inbound[1:]
            inv_patch[shelf] = item
            action = f"STORE {item} {shelf}"
            reasoning = f"Inbound {item}; storing on empty {shelf}."
    elif pending:
        item = pending[0]
        shelf = _find_item(inventory, item)
        if shelf is None:
            action = "WAIT"
            reasoning = f"{item} is ordered but not on a shelf yet; WAIT."
        else:
            inventory[shelf] = None
            inv_patch[shelf] = None
            pending = pending[1:]
            shipped = shipped + [item]
            action = f"SHIP {item} {shelf}"
            reasoning = f"Pending order {item} sits on {shelf}; shipping."
    else:
        idle = "Warehouse is idle" in observation or "No new events" in observation
        if observation.startswith("ERROR:") or "cannot DONE" in observation:
            action = "WAIT"
            reasoning = "Environment rejected the last action; WAIT."
        elif idle:
            action = "DONE"
            reasoning = "Inbound empty, pending empty, warehouse idle; DONE."
        else:
            action = "WAIT"
            reasoning = "Caught up on known work; WAIT for the next event."

    if inv_patch:
        patch["inventory"] = inv_patch
    patch["inbound"] = inbound
    patch["pending_orders"] = pending
    patch["shipped"] = shipped
    patch["last_action"] = action

    payload = {"state_patch": patch, "action": action}
    return (
        reasoning
        + "\n```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```\n"
    )


_HISTORY_EVENT = re.compile(
    r"Shipment arrived containing (item_\d{2})"
    r"|Customer ordered (item_\d{2})"
    r"|Another worker moved (item_\d{2}) from (shelf_\d{2}) to (shelf_\d{2})"
    r"|Success: Stored (item_\d{2}) on (shelf_\d{2})"
    r"|Success: Shipped (item_\d{2}) from (shelf_\d{2})"
    r"|Cycle count: (item_\d{2}) is not on (shelf_\d{2})\. Floor scanner: item_\d{2} is on (shelf_\d{2})"
)


def warehouse_history_policy(prompt: str) -> str:
    """ReAct policy: reconstruct the world from the transcript in chronological order."""
    inventory = empty_inventory()
    inbound: list[str] = []
    pending: list[str] = []
    shipped: list[str] = []

    for match in _HISTORY_EVENT.finditer(prompt):
        g = match.groups()
        if g[0]:
            item = g[0]
            if item not in inbound and item not in inventory.values() and item not in shipped:
                inbound.append(item)
        elif g[1]:
            item = g[1]
            if item not in pending and item not in shipped:
                pending.append(item)
        elif g[2]:
            item, src, dst = g[2], g[3], g[4]
            inventory[src] = None
            inventory[dst] = item
        elif g[5]:
            item, shelf = g[5], g[6]
            if item in inbound:
                inbound.remove(item)
            inventory[shelf] = item
        elif g[7]:
            item, shelf = g[7], g[8]
            inventory[shelf] = None
            if item in pending:
                pending.remove(item)
            if item not in shipped:
                shipped.append(item)
        elif g[9]:
            item, src, dst = g[9], g[10], g[11]
            inventory[src] = None
            inventory[dst] = item

    if inbound:
        item = inbound[0]
        shelf = _first_empty(inventory)
        if shelf:
            thought = f"History shows {item} inbound; store on {shelf}."
            action = f"STORE {item} {shelf}"
        else:
            thought = "No empty shelf."
            action = "WAIT"
    elif pending:
        item = pending[0]
        shelf = _find_item(inventory, item)
        if shelf:
            thought = f"History shows {item} on {shelf}; ship it."
            action = f"SHIP {item} {shelf}"
        else:
            thought = f"{item} ordered but not located yet."
            action = "WAIT"
    else:
        idle = "Warehouse is idle" in prompt or "No new events" in prompt
        if idle:
            thought = "Transcript shows no remaining work and the warehouse is idle."
            action = "DONE"
        else:
            thought = "Caught up; waiting for the next event."
            action = "WAIT"

    # Guard: only emit grammar-valid actions.
    if parse_warehouse_action(action) is None:
        action = "WAIT"
    return f"{thought}\nAction: {action}"


def repoops_skillstate_policy(prompt: str) -> str:
    from skillstate.skills.repoops.env import FIX_MARKER, LOG_SNIPPET

    state, observation = parse_skill_state_prompt(prompt)
    branches = {
        name: dict(meta) for name, meta in (state.get("branches") or {}).items()
    }
    prs = [dict(p) for p in (state.get("prs") or [])]
    tickets = list(state.get("tickets") or [])
    step = int(state.get("step") or 0) + 1
    patch: dict[str, Any] = {"step": step}

    opened = re.search(r"opened PR #(\d+)", observation)
    if opened:
        pr_id = int(opened.group(1))
        if not any(p.get("id") == pr_id for p in prs):
            prs.append(
                {"id": pr_id, "source": "feat-1", "target": "main", "status": "open"}
            )

    if "Ticket OP-1" in observation and "OP-1" not in tickets:
        tickets.append("OP-1")
        patch["tickets"] = tickets

    feat = "feat-1"
    action = "WAIT"
    reasoning = "Scripted repoops policy."

    if feat not in branches:
        action = f"BRANCH {feat}"
        reasoning = "Need a feature branch."
        main_file = branches.get("main", {}).get("file", "print('hello')\n")
        patch["branches"] = {feat: {"file": main_file, "ci": "unknown"}}
    else:
        feat_state = branches[feat]
        file_text = feat_state.get("file", "")
        ci = feat_state.get("ci", "unknown")
        has_pr = any(p.get("source") == feat and p.get("status") == "open" for p in prs)
        merged = any(p.get("status") == "merged" for p in prs)
        if LOG_SNIPPET.strip() not in file_text:
            action = f"COMMIT {feat} add_logging"
            reasoning = "Commit the logging change."
            new_file = file_text.rstrip("\n") + "\n" + LOG_SNIPPET
            patch["branches"] = {feat: {"file": new_file, "ci": "pending"}}
        elif FIX_MARKER.strip() not in file_text:
            action = f"COMMIT {feat} fix_ci"
            reasoning = "Commit the CI fix."
            new_file = file_text.rstrip("\n") + "\n" + FIX_MARKER
            patch["branches"] = {feat: {"file": new_file, "ci": "pending"}}
        elif not has_pr and not merged:
            action = f"CREATE_PR {feat}"
            reasoning = "Open the pull request."
            next_id = 1 + max([p.get("id", 0) for p in prs], default=0)
            patch["prs"] = prs + [
                {"id": next_id, "source": feat, "target": "main", "status": "open"}
            ]
        elif ci != "pass" and not merged:
            action = f"RUN_CI {feat}"
            reasoning = "Run CI on the feature branch."
            ok = LOG_SNIPPET.strip() in file_text and FIX_MARKER.strip() in file_text
            patch["branches"] = {feat: {**feat_state, "ci": "pass" if ok else "fail"}}
        elif has_pr:
            pr = next(p for p in prs if p.get("source") == feat and p.get("status") == "open")
            action = f"MERGE {pr['id']}"
            reasoning = "CI is green; merge."
            patch["prs"] = [
                {**p, "status": "merged"} if p.get("id") == pr["id"] else p for p in prs
            ]
            patch["branches"] = {"main": {"file": file_text, "ci": "pass"}}
            patch["tickets"] = [t for t in tickets if t != "OP-1"]
        else:
            action = "DONE"
            reasoning = "PR merged; DONE."

    if "CI passed" in observation:
        feat_state = branches.get(feat, {})
        patch.setdefault("branches", {})
        if feat in branches:
            patch["branches"][feat] = {**feat_state, "ci": "pass"}

    patch["last_action"] = action
    payload = {"state_patch": patch, "action": action}
    return reasoning + "\n```json\n" + json.dumps(payload) + "\n```\n"


_NAME = re.compile(r"my name is ([A-Za-z][\w-]{0,31})", re.IGNORECASE)


def chat_skillstate_policy(prompt: str) -> str:
    """Two-turn canned chat policy for offline /chat and pytest."""
    state, observation = parse_skill_state_prompt(prompt)
    facts = [str(f) for f in (state.get("facts") or []) if str(f).strip()]
    goal = str(state.get("goal") or "")
    questions = [str(q) for q in (state.get("open_questions") or []) if str(q).strip()]
    decisions = [str(d) for d in (state.get("decisions") or []) if str(d).strip()]
    tone = str(state.get("tone") or "direct")

    if observation.startswith("VALIDATOR ERROR"):
        action = "SAY Please say that again — I failed to format the last reply."
        reasoning = "Validator rejected the last payload; emit a legal SAY."
    elif observation.startswith("TOOL CALC:"):
        action = f"SAY {observation.split(':', 1)[1].strip()}."
        reasoning = "Report the calculator result."
    elif observation.startswith("TOOL TIME:"):
        action = f"SAY The local time is {observation.split(':', 1)[1].strip()}."
        reasoning = "Report the clock result."
    elif observation.startswith("TOOL HASH:"):
        action = f"SAY SHA-256 {observation.split(':', 1)[1].strip()}."
        reasoning = "Report the hash result."
    elif observation.startswith("TOOL ERROR"):
        action = f"SAY The tool failed: {observation}."
        reasoning = "Surface the tool error."
    elif re.search(r"\b(what time|time is it|current time)\b", observation, re.I):
        action = "TIME"
        reasoning = "Need the clock; call TIME."
    elif (calc := re.search(
        r"(?:calc(?:ulate)?|what is)\s+([0-9.+\-*/() ]{1,80})\s*\??\s*$",
        observation,
        re.I,
    )):
        action = f"CALC {calc.group(1).strip()}"
        reasoning = "Need arithmetic; call CALC."
    elif (hashed := re.search(r"\bhash\s+(.+)$", observation, re.I)):
        action = f"HASH {hashed.group(1).strip()}"
        reasoning = "Need a digest; call HASH."
    elif re.search(r"\b(goodbye|that's enough|that is enough|done)\b", observation, re.I):
        action = "DONE Glad we could talk."
        reasoning = "User asked to stop; DONE."
        questions = []
        decisions = decisions + ["ended_by_user"]
    else:
        named = _NAME.search(observation)
        if named:
            name = named.group(1)
            facts = [f for f in facts if not f.lower().startswith("user_name=")]
            facts.append(f"user_name={name}")
        want = re.search(r"\b(?:want|need|looking for)\b(.{0,60})", observation, re.I)
        if want and not goal:
            goal = want.group(1).strip(" .,:;")[:80]
        remembered = next(
            (f.split("=", 1)[1] for f in facts if f.lower().startswith("user_name=")),
            "",
        )
        if remembered and re.search(r"\b(what(?:'s| is) my name)\b", observation, re.I):
            action = f"SAY Your name is {remembered}."
            reasoning = "O asks for a fact already in Σ; answer from Σ."
        else:
            action = (
                f"SAY I have that. Tell me if you want to change anything "
                f"or say that's enough."
            )
            reasoning = "Record facts from O; do not keep the utterance itself."
        questions = []

    patch: dict[str, Any] = {
        "goal": goal,
        "facts": facts,
        "open_questions": questions,
        "decisions": decisions,
        "tone": tone,
        "last_action": action,
    }
    payload = {"state_patch": patch, "action": action}
    return reasoning + "\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n"
