from __future__ import annotations

from skillstate.skills.warehouse.env import (
    WarehouseEnv,
    generate_scenario,
    parse_warehouse_action,
)
from skillstate.skills.warehouse.schema import SHELF_IDS, empty_inventory


def test_parse_accepts_strict_grammar():
    a = parse_warehouse_action("STORE item_00 shelf_03")
    assert a is not None and a.op == "STORE" and a.item == "item_00"
    assert parse_warehouse_action("SHIP item_00 shelf_00").op == "SHIP"
    assert parse_warehouse_action("MOVE item_00 shelf_00 shelf_01").op == "MOVE"
    assert parse_warehouse_action("WAIT").op == "WAIT"
    assert parse_warehouse_action("DONE").op == "DONE"


def test_parse_rejects_junk():
    assert parse_warehouse_action("store-item") is None
    assert parse_warehouse_action("STORE item_00") is None
    assert parse_warehouse_action("STORE item_0 shelf_00") is None
    assert parse_warehouse_action("FLY item_00 shelf_00") is None
    assert parse_warehouse_action("") is None
    assert parse_warehouse_action("STORE item_00 shelf_00 NOW") is None


def test_store_ship_happy_path():
    env = WarehouseEnv(seed=1, compact=True, horizon=8)
    obs = env.reset()
    assert "item_00" in obs
    obs, done, info = env.step("STORE item_00 shelf_00")
    assert info["valid"]
    assert env.shelves["shelf_00"] == "item_00"
    assert "item_01" in obs
    env.step("STORE item_01 shelf_01")
    env.step("SHIP item_00 shelf_00")
    env.step("SHIP item_01 shelf_01")
    assert env.success()
    obs, done, info = env.step("DONE")
    assert done and info["success"]


def test_store_occupied_rejected():
    env = WarehouseEnv(seed=1, compact=True, horizon=8)
    env.reset()
    env.step("STORE item_00 shelf_00")
    obs, _, info = env.step("STORE item_01 shelf_00")
    assert not info["valid"]
    assert "occupied" in obs.lower()
    assert env.shelves["shelf_00"] == "item_00"


def test_ship_wrong_shelf_rejected():
    env = WarehouseEnv(seed=1, compact=True, horizon=8)
    env.reset()
    env.step("STORE item_00 shelf_00")
    env.step("STORE item_01 shelf_01")
    obs, _, info = env.step("SHIP item_00 shelf_01")
    assert not info["valid"]


def test_move_and_drift_mutate_ground_truth():
    env = WarehouseEnv(seed=0, compact=True, horizon=8)
    env.reset()
    env.inbound.append("item_09")
    ok, valid = env._store("item_09", "shelf_05")
    assert valid
    obs, _, info = env.step("MOVE item_09 shelf_05 shelf_08")
    assert info["valid"]
    assert env.shelves["shelf_05"] is None
    assert env.shelves["shelf_08"] == "item_09"


def test_seed_reproducible_events():
    a = generate_scenario(7, horizon=36)
    b = generate_scenario(7, horizon=36)
    c = generate_scenario(8, horizon=36)
    assert a == b
    assert a != c
    kinds = [e.kind for e in a]
    assert "receive" in kinds and "order" in kinds


def test_short_horizon_still_has_orders():
    events = generate_scenario(42, horizon=12)
    assert any(e.kind == "receive" for e in events)
    assert any(e.kind == "order" for e in events)
    assert len(events) <= 12


def test_compact_is_four_events():
    events = generate_scenario(1, compact=True)
    assert [e.kind for e in events] == ["receive", "receive", "order", "order"]


def test_empty_inventory_has_24_null_shelves():
    inv = empty_inventory()
    assert list(inv) == list(SHELF_IDS)
    assert all(v is None for v in inv.values())
