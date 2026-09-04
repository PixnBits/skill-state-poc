"""Warehouse execution-state schema. extra='forbid' is load-bearing."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

N_SHELVES = 24
SHELF_IDS: tuple[str, ...] = tuple(f"shelf_{i:02d}" for i in range(N_SHELVES))
SHELF_SET = frozenset(SHELF_IDS)
ITEM_RE = re.compile(r"^item_\d{2}$")
SHELF_RE = re.compile(r"^shelf_\d{2}$")


def empty_inventory() -> dict[str, str | None]:
    return {shelf: None for shelf in SHELF_IDS}


class WarehouseState(BaseModel):
    """Σ for the warehouse skill.

    ``inbound`` is not in the paper's 500-shelf sketch; it is required for Σ
    to be a sufficient statistic when a shipment is not stored on the same
    step it arrives (otherwise O_t is discarded and the item is forgotten).
    """

    model_config = ConfigDict(extra="forbid")

    inventory: dict[str, str | None]
    inbound: list[str] = Field(default_factory=list)
    pending_orders: list[str] = Field(default_factory=list)
    shipped: list[str] = Field(default_factory=list)
    last_action: str = ""
    step: int = 0

    @field_validator("inventory")
    @classmethod
    def inventory_is_exactly_the_shelves(
        cls, value: dict[str, str | None]
    ) -> dict[str, str | None]:
        extra = sorted(set(value.keys()) - SHELF_SET)
        if extra:
            raise ValueError(f"unknown shelves in inventory: {extra}")
        # Exception to RFC 7396, documented in merge.py / test_merge.py:
        # ``{"inventory": {"shelf_00": null}}`` deletes the key in ⊕, but a
        # missing known shelf means "empty", not "undeclared". Re-fill so Σ
        # remains a 24-key map. Unknown top-level keys still fail extra=forbid.
        filled = empty_inventory()
        filled.update(value)
        for shelf, item in filled.items():
            if item is not None and not ITEM_RE.match(item):
                raise ValueError(f"{shelf} holds invalid item id {item!r}")
        occupied = [item for item in filled.values() if item is not None]
        if len(occupied) != len(set(occupied)):
            raise ValueError("the same item cannot occupy two shelves")
        return filled

    @field_validator("inbound", "pending_orders", "shipped")
    @classmethod
    def item_lists(cls, value: list[str]) -> list[str]:
        for item in value:
            if not ITEM_RE.match(item):
                raise ValueError(f"invalid item id {item!r}")
        return value

    @field_validator("step")
    @classmethod
    def step_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("step must be >= 0")
        return value

    @model_validator(mode="after")
    def no_duplicate_item_locations(self) -> WarehouseState:
        on_shelves = {v for v in self.inventory.values() if v is not None}
        overlap = on_shelves.intersection(self.inbound)
        if overlap:
            raise ValueError(f"items both inbound and on a shelf: {sorted(overlap)}")
        return self


def initial_warehouse_state() -> dict[str, Any]:
    return WarehouseState(
        inventory=empty_inventory(),
        inbound=[],
        pending_orders=[],
        shipped=[],
        last_action="",
        step=0,
    ).model_dump(mode="json")
