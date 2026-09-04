"""Dictionary merge operator ⊕ from SKILL.state (arXiv:2608.26263 §3.2).

Σ_{t+1} = Σ_t ⊕ ΔΣ_t

Semantics follow JSON Merge Patch (RFC 7396) with explicit null-deletion:

- A JSON ``null`` in the patch **deletes** that key from the object.
- Nested objects are merged key-wise (not replaced wholesale).
- Arrays and scalars are replaced (not concatenated / element-wise merged).
- The operator itself is schema-agnostic. Pass ``allowed_top_keys`` to refuse
  undeclared top-level keys at merge time. Independently, pydantic models in
  this PoC use ``extra="forbid"`` so a merged Σ that grew a new top-level
  field is rejected even if the allow-list check is skipped.

This PoC does **not** use ``extra="ignore"``. Unknown fields are errors.

Warehouse exception (the only one): JSON null inside ``inventory`` deletes
that shelf key under RFC 7396, but an empty shelf is still a declared slot.
``WarehouseState`` re-fills missing ``shelf_NN`` keys as ``None``. Unknown
shelf ids and unknown top-level keys still fail.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MergeError(ValueError):
    """Patch is structurally unusable (unknown keys, wrong type, …)."""


def deep_merge(
    base: Any,
    patch: Any,
    *,
    allowed_top_keys: set[str] | frozenset[str] | None = None,
) -> Any:
    """Return ``base ⊕ patch`` without mutating either argument.

    ``allowed_top_keys`` is enforced only at the top-level call (the keys of
    ``patch`` itself), matching “do not invent undeclared top-level keys”.
    Nested keys (e.g. individual shelves) are the schema’s job.
    """
    if allowed_top_keys is not None:
        if not isinstance(patch, dict):
            raise MergeError(
                f"state_patch must be an object, got {type(patch).__name__}"
            )
        unknown = set(patch.keys()) - set(allowed_top_keys)
        if unknown:
            hint = ""
            if any(str(k).startswith("shelf_") for k in unknown):
                hint = (
                    " Hint: shelf_* keys belong under 'inventory', e.g. "
                    '{"inventory": {"shelf_00": "item_00"}}.'
                )
            raise MergeError(
                "undeclared top-level keys in state_patch: "
                f"{sorted(unknown)}; allowed: {sorted(allowed_top_keys)}."
                f"{hint}"
            )
    return _merge(base, patch)


def _merge(base: Any, patch: Any) -> Any:
    if patch is None:
        return None
    if not isinstance(patch, dict):
        return deepcopy(patch)
    if not isinstance(base, dict):
        # Patch is an object but base is a scalar/list/None: replace.
        out: dict[str, Any] = {}
        for key, value in patch.items():
            if value is None:
                continue
            out[key] = _merge({}, value) if isinstance(value, dict) else deepcopy(value)
        return out
    out = deepcopy(base)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        elif isinstance(value, dict):
            out[key] = _merge({}, value)
        else:
            out[key] = deepcopy(value)
    return out
