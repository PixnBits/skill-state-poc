"""Deterministic 24-shelf warehouse (paper Env 1, scaled down from 500).

Ground-truth world lives here. The agent's Σ is a separate dict owned by the
runtime. Observations are event alerts plus the result of the last action —
never a dump of the full warehouse, never prior events.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from skillstate.skills.warehouse.schema import (
    ITEM_RE,
    N_SHELVES,
    SHELF_IDS,
    SHELF_RE,
    empty_inventory,
)

Op = Literal["STORE", "SHIP", "MOVE", "WAIT", "DONE"]


@dataclass(frozen=True)
class WarehouseAction:
    op: Op
    item: str | None = None
    shelf: str | None = None
    from_shelf: str | None = None
    to_shelf: str | None = None
    raw: str = ""


@dataclass(frozen=True)
class ScheduledEvent:
    kind: Literal["receive", "order", "drift"]
    item: str | None = None


def parse_warehouse_action(command: str) -> WarehouseAction | None:
    """Strict grammar. Unknown opcodes / arity → None (validator failure)."""
    if command is None:
        return None
    parts = command.strip().split()
    if not parts:
        return None
    op = parts[0].upper()
    raw = " ".join([op, *parts[1:]]) if parts[1:] else op
    if op == "WAIT" and len(parts) == 1:
        return WarehouseAction(op="WAIT", raw="WAIT")
    if op == "DONE" and len(parts) == 1:
        return WarehouseAction(op="DONE", raw="DONE")
    if op == "STORE" and len(parts) == 3:
        item, shelf = parts[1], parts[2]
        if ITEM_RE.match(item) and SHELF_RE.match(shelf):
            return WarehouseAction(op="STORE", item=item, shelf=shelf, raw=raw)
        return None
    if op == "SHIP" and len(parts) == 3:
        item, shelf = parts[1], parts[2]
        if ITEM_RE.match(item) and SHELF_RE.match(shelf):
            return WarehouseAction(op="SHIP", item=item, shelf=shelf, raw=raw)
        return None
    if op == "MOVE" and len(parts) == 4:
        item, src, dst = parts[1], parts[2], parts[3]
        if ITEM_RE.match(item) and SHELF_RE.match(src) and SHELF_RE.match(dst):
            return WarehouseAction(
                op="MOVE", item=item, from_shelf=src, to_shelf=dst, raw=raw
            )
        return None
    return None


def generate_scenario(
    seed: int,
    *,
    n_shelves: int = N_SHELVES,
    horizon: int = 36,
    compact: bool = False,
) -> list[ScheduledEvent]:
    """Seeded event list. Packed at the front of the horizon so the agent can finish.

    compact=True produces the 5-step unit-test episode:
      receive, receive, order, order, (idle → DONE)
    """
    if compact:
        return [
            ScheduledEvent("receive", "item_00"),
            ScheduledEvent("receive", "item_01"),
            ScheduledEvent("order", "item_00"),
            ScheduledEvent("order", "item_01"),
        ]
    rng = random.Random(seed)
    # Leave headroom so the agent can STORE each receive, SHIP each order, DONE.
    n_items = max(2, min(8, max(2, (horizon - 2) // 3)))
    n_orders = max(1, min(n_items, max(1, (horizon - 2) // 4)))
    n_drift = 1 if horizon >= 16 else (1 if horizon >= 14 and rng.random() < 0.5 else 0)
    items = [f"item_{i:02d}" for i in range(n_items)]
    order_items = list(items[:n_orders])
    rng.shuffle(order_items)

    events: list[ScheduledEvent] = []
    received: list[str] = []
    drifts_left = n_drift

    for item in items:
        events.append(ScheduledEvent("receive", item))
        received.append(item)
        if len(received) >= 2 and order_items and rng.random() < 0.55:
            events.append(ScheduledEvent("order", order_items.pop(0)))
        if len(received) >= 3 and drifts_left and rng.random() < 0.35:
            events.append(ScheduledEvent("drift", None))
            drifts_left -= 1

    events.extend(ScheduledEvent("order", nxt) for nxt in order_items)
    events.extend(ScheduledEvent("drift", None) for _ in range(drifts_left))
    return events


@dataclass
class WarehouseEnv:
    seed: int = 42
    n_shelves: int = N_SHELVES
    horizon: int = 36
    compact: bool = False
    t: int = 0
    shelves: dict[str, str | None] = field(default_factory=empty_inventory)
    inbound: list[str] = field(default_factory=list)
    pending_orders: list[str] = field(default_factory=list)
    shipped: list[str] = field(default_factory=list)
    events: list[ScheduledEvent] = field(default_factory=list)
    ordered_items: list[str] = field(default_factory=list)
    last_info: dict[str, Any] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))
    n_actionable_events: int = 0
    n_valid_actions: int = 0
    n_invalid_actions: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed + 17)
        self.events = generate_scenario(
            self.seed,
            n_shelves=self.n_shelves,
            horizon=self.horizon,
            compact=self.compact,
        )
        self.ordered_items = [e.item for e in self.events if e.kind == "order" and e.item]
        self.n_actionable_events = len(self.events)
        self.shelves = empty_inventory()
        if self.n_shelves != N_SHELVES:
            # PoC schema is fixed at 24; keep env aligned.
            raise ValueError("this PoC warehouse is fixed at 24 shelves")

    def reset(self) -> str:
        self.t = 0
        self.shelves = empty_inventory()
        self.inbound = []
        self.pending_orders = []
        self.shipped = []
        self.n_valid_actions = 0
        self.n_invalid_actions = 0
        self.last_info = {}
        briefing = (
            f"Shift start. {self.n_shelves} shelves (shelf_00..shelf_23), "
            f"horizon {self.horizon} steps, seed {self.seed}. "
            "Handle every shipment and customer order, recover from worker-move "
            "alerts, then DONE."
        )
        return _compose_observation(briefing, self._fire_event_at(self.t))

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        parsed = parse_warehouse_action(action)
        if parsed is None:
            # Grammar failures are the runtime's job; if we get here, treat as invalid.
            result = f"ERROR: action does not match warehouse grammar: {action!r}"
            self.n_invalid_actions += 1
            valid = False
        else:
            result, valid = self._apply(parsed)
            if valid:
                self.n_valid_actions += 1
            else:
                self.n_invalid_actions += 1

        self.t += 1
        observation = _compose_observation(result, self._fire_event_at(self.t))
        succeeded = self.success()
        done = False
        if parsed is not None and parsed.op == "DONE" and succeeded:
            done = True
        if self.t >= self.horizon:
            done = True
        info = {
            "valid": valid,
            "success": succeeded,
            "gt": self.snapshot(),
            "t": self.t,
        }
        self.last_info = info
        return observation, done, info

    def success(self) -> bool:
        if not self.ordered_items:
            return False
        if self.pending_orders:
            return False
        if self.inbound:
            return False
        if set(self.shipped) != set(self.ordered_items):
            return False
        if len(self.shipped) != len(set(self.shipped)):
            return False
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "inventory": dict(self.shelves),
            "inbound": list(self.inbound),
            "pending_orders": list(self.pending_orders),
            "shipped": list(self.shipped),
            "ordered_items": list(self.ordered_items),
            "t": self.t,
            "n_events": len(self.events),
            "n_valid_actions": self.n_valid_actions,
            "n_invalid_actions": self.n_invalid_actions,
            "n_actionable_events": self.n_actionable_events,
        }

    def score(self) -> float:
        if self.n_actionable_events == 0:
            return 1.0 if self.success() else 0.0
        # Paper: successful actions / actionable events. Clamp to [0, 1].
        return min(1.0, self.n_valid_actions / self.n_actionable_events)

    def _fire_event_at(self, step: int) -> str | None:
        if step < 0 or step >= len(self.events):
            return None
        event = self.events[step]
        if event.kind == "receive":
            assert event.item is not None
            if event.item not in self.inbound and event.item not in self.shelves.values():
                self.inbound.append(event.item)
            return f"Shipment arrived containing {event.item}."
        if event.kind == "order":
            assert event.item is not None
            if event.item not in self.pending_orders and event.item not in self.shipped:
                self.pending_orders.append(event.item)
            return f"Customer ordered {event.item}."
        if event.kind == "drift":
            return self._apply_drift()
        return None

    def _apply_drift(self) -> str:
        occupied = [(s, item) for s, item in self.shelves.items() if item is not None]
        empty = [s for s, item in self.shelves.items() if item is None]
        if not occupied or not empty:
            return "ALERT: another worker shuffled empty pallets (no inventory change)."
        src, item = occupied[self._rng.randrange(len(occupied))]
        dst = empty[self._rng.randrange(len(empty))]
        self.shelves[src] = None
        self.shelves[dst] = item
        return f"ALERT: Another worker moved {item} from {src} to {dst}."

    def _apply(self, action: WarehouseAction) -> tuple[str, bool]:
        if action.op == "WAIT":
            return "OK: waited.", True
        if action.op == "DONE":
            if self.success():
                return "OK: shift complete. All orders shipped.", True
            return (
                "ERROR: cannot DONE. "
                f"inbound={self.inbound} pending_orders={self.pending_orders} "
                f"shipped={self.shipped} expected_orders={self.ordered_items}",
                False,
            )
        if action.op == "STORE":
            return self._store(action.item, action.shelf)  # type: ignore[arg-type]
        if action.op == "SHIP":
            return self._ship(action.item, action.shelf)  # type: ignore[arg-type]
        if action.op == "MOVE":
            return self._move(action.item, action.from_shelf, action.to_shelf)  # type: ignore[arg-type]
        return f"ERROR: unknown op {action.op}", False

    def _store(self, item: str, shelf: str) -> tuple[str, bool]:
        if shelf not in self.shelves:
            return f"ERROR: unknown shelf {shelf}.", False
        if item not in self.inbound:
            return f"ERROR: {item} is not on the inbound dock ({self.inbound}).", False
        if self.shelves[shelf] is not None:
            return (
                f"ERROR: {shelf} is occupied by {self.shelves[shelf]}. "
                "Choose an empty shelf.",
                False,
            )
        self.inbound.remove(item)
        self.shelves[shelf] = item
        return f"Success: Stored {item} on {shelf}.", True

    def _ship(self, item: str, shelf: str) -> tuple[str, bool]:
        if shelf not in self.shelves:
            return f"ERROR: unknown shelf {shelf}.", False
        if self.shelves[shelf] != item:
            return (
                f"ERROR: {shelf} holds {self.shelves[shelf]!r}, not {item}.",
                False,
            )
        if item not in self.pending_orders:
            return f"ERROR: {item} is not a pending order ({self.pending_orders}).", False
        self.shelves[shelf] = None
        self.pending_orders.remove(item)
        self.shipped.append(item)
        return f"Success: Shipped {item} from {shelf}.", True

    def _move(self, item: str, src: str, dst: str) -> tuple[str, bool]:
        if src not in self.shelves or dst not in self.shelves:
            return "ERROR: unknown shelf in MOVE.", False
        if self.shelves[src] != item:
            return f"ERROR: {src} holds {self.shelves[src]!r}, not {item}.", False
        if self.shelves[dst] is not None:
            return f"ERROR: {dst} is occupied by {self.shelves[dst]}.", False
        self.shelves[src] = None
        self.shelves[dst] = item
        return f"Success: Moved {item} from {src} to {dst}.", True


_SHIPMENT_RE = re.compile(r"Shipment arrived containing (item_\d{2})")
_ORDER_RE = re.compile(r"Customer ordered (item_\d{2})")
_DRIFT_RE = re.compile(
    r"Another worker moved (item_\d{2}) from (shelf_\d{2}) to (shelf_\d{2})"
)
_STORED_RE = re.compile(r"Stored (item_\d{2}) on (shelf_\d{2})")
_SHIPPED_RE = re.compile(r"Shipped (item_\d{2}) from (shelf_\d{2})")
_MOVED_RE = re.compile(r"Moved (item_\d{2}) from (shelf_\d{2}) to (shelf_\d{2})")


def _join(*parts: str | None) -> str:
    return "\n".join(p for p in parts if p)


def _compose_observation(result: str | None, event_text: str | None) -> str:
    parts = [p for p in (result, event_text) if p]
    if event_text is None:
        parts.append("No new events. Warehouse is idle.")
    return "\n".join(parts)
