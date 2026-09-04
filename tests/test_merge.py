"""⊕ operator: nested overwrite, null-delete, extra-key refusal."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from skillstate.merge import MergeError, deep_merge
from skillstate.schemas import PatchValidationError, apply_validated_patch
from skillstate.skills.warehouse.schema import WarehouseState, empty_inventory, initial_warehouse_state


def test_nested_overwrite_keeps_siblings():
    base = {"inventory": {"shelf_00": "item_01", "shelf_01": "item_02"}, "step": 1}
    patch = {"inventory": {"shelf_00": "item_09"}}
    out = deep_merge(base, patch)
    assert out["inventory"]["shelf_00"] == "item_09"
    assert out["inventory"]["shelf_01"] == "item_02"
    assert out["step"] == 1
    assert base["inventory"]["shelf_00"] == "item_01"  # no mutation


def test_null_deletes_key():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]}
    out = deep_merge(base, {"a": None, "b": {"c": None}})
    assert "a" not in out
    assert "c" not in out["b"]
    assert out["b"]["d"] == 3
    assert out["e"] == [1, 2]


def test_null_deletes_nested_key_rfc7396():
    # Paper example `inventory.shelf_k: null` deletes the key in ⊕.
    # WarehouseState re-fills known shelves (see test_warehouse_null_shelf_is_empty_not_missing).
    base = {"inventory": {"shelf_00": "item_01", "shelf_01": "item_02"}}
    out = deep_merge(base, {"inventory": {"shelf_00": None}})
    assert "shelf_00" not in out["inventory"]
    assert out["inventory"]["shelf_01"] == "item_02"


def test_list_replaced_not_concatenated():
    out = deep_merge({"pending_orders": ["a", "b"]}, {"pending_orders": ["c"]})
    assert out["pending_orders"] == ["c"]


def test_empty_patch_is_identity():
    base = {"x": {"y": 1}}
    out = deep_merge(base, {})
    assert out == base
    assert out is not base


def test_refuses_undeclared_top_level_keys():
    with pytest.raises(MergeError, match="undeclared"):
        deep_merge({"a": 1}, {"b": 2}, allowed_top_keys={"a"})


def test_scalar_overwrites_object():
    assert deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}


def test_object_overwrites_scalar():
    assert deep_merge({"a": 5}, {"a": {"b": 1}}) == {"a": {"b": 1}}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    b: str | None = None


def test_apply_validated_patch_rejects_unknown_top_level():
    with pytest.raises(PatchValidationError, match="undeclared"):
        apply_validated_patch({"a": 1, "b": None}, {"zzz": 1}, Strict)


def test_apply_validated_patch_accepts_partial():
    out = apply_validated_patch({"a": 1, "b": "x"}, {"b": "y"}, Strict)
    assert out == {"a": 1, "b": "y"}


def test_apply_validated_patch_null_deletes_optional():
    out = apply_validated_patch({"a": 1, "b": "x"}, {"b": None}, Strict)
    assert out["a"] == 1
    assert out["b"] is None


def test_warehouse_null_shelf_is_empty_not_missing():
    """Paper example: ``inventory.shelf_k: null`` means the shelf is empty.

    RFC 7396 would drop the key; the warehouse schema fills missing known
    shelves back in as None so Σ stays a 24-key map. Unknown shelf ids still
    fail extra-key checks on the inventory dict.
    """
    state = initial_warehouse_state()
    state["inventory"]["shelf_00"] = "item_01"
    # Direct merge drops the key...
    merged = deep_merge(state, {"inventory": {"shelf_00": None}})
    assert "shelf_00" not in merged["inventory"]
    # ...but apply_validated_patch rehydrates it via the schema.
    out = apply_validated_patch(state, {"inventory": {"shelf_00": None}}, WarehouseState)
    assert out["inventory"]["shelf_00"] is None
    assert set(out["inventory"]) == set(empty_inventory())
