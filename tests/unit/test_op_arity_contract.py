"""The catalog's `min_arity` must match the reference op's real signature.

W1.2 found 41 of the 106 unclassified ops failing an arity probe. Most were not
catalog bugs: the trailing positional was an ATTRIBUTE (a shape, a pad width, a
dtype) that `_POSITIONAL_ATTR_PARAMS` had not declared, so the frontend counted
it as a tensor operand. That is the same gap that blocks Graph IR emission for
structural view ops.

Two lessons are encoded here:

  * **Operands are not parameters.** A naive `len(positional params)` check
    reports ~30 mismatches, almost all false. The comparison must add the
    declared positional attributes.
  * **Keyword-only attributes have no declaration site.** `softcap(x, *, cap)`
    and `attn_sliding_window(q, k, v, *, window)` take required keyword-only
    ATTRIBUTES, and `_POSITIONAL_ATTR_PARAMS` governs positional ones only. So
    this gate scopes itself to positional parameters; flagging the rest would
    report drift the catalog cannot currently express. That missing vocabulary
    is a real W1 item, recorded rather than papered over.

Both mistakes were made while building this, which is why the check states
them explicitly rather than just asserting a number.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from tessera.compiler.graph_ir import _POSITIONAL_ATTR_PARAMS
from tessera.compiler.op_catalog import _SPECS


def _required_positional_count(fn):
    """Required POSITIONAL parameters, or None if variadic.

    Deliberately excludes required keyword-only parameters. `softcap(x, *,
    cap)`, `attn_sliding_window(q, k, v, *, window)` and friends take required
    keyword-only ATTRIBUTES, and `_POSITIONAL_ATTR_PARAMS` — as its name says —
    only governs positional ones. There is no declaration site for keyword-only
    attributes today; counting them here would report drift the catalog has no
    way to resolve. That gap is real and recorded in the W1 inventory, but it is
    a separate vocabulary item, not something this gate can enforce.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    count = 0
    for param in signature.parameters.values():
        if param.kind is param.VAR_POSITIONAL:
            return None  # unbounded; arity is not comparable
        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            continue
        if param.default is param.empty:
            count += 1
    return count


def test_catalog_arity_matches_reference_signatures():
    """`min_arity` + declared positional attributes must equal required params.

    A mismatch means one of three real problems: the catalog miscounts the
    operands, a positional attribute is undeclared (so the frontend will treat
    it as a tensor), or the reference signature drifted from the contract.
    """
    from tessera import ops

    mismatches = []
    seen: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in _SPECS:
            if spec.graph_name in seen:
                continue
            seen.add(spec.graph_name)
            fn = getattr(ops, spec.public_name, None)
            if fn is None:
                continue
            required = _required_positional_count(fn)
            if required is None:
                continue
            attrs = len(_POSITIONAL_ATTR_PARAMS.get(spec.graph_name, ()))
            # Defaulted operands (flash_attn's K/V) legitimately make the
            # function accept fewer than the semantic arity, so only flag the
            # case where the function REQUIRES more than the contract declares.
            if required > spec.min_arity + attrs:
                mismatches.append(
                    f"{spec.public_name}: catalog min_arity={spec.min_arity} "
                    f"+ {attrs} declared attrs, but the reference requires "
                    f"{required} parameters"
                )

    assert not mismatches, (
        "catalog arity disagrees with the reference signatures:\n  "
        + "\n  ".join(mismatches)
        + "\nEither declare the trailing positional in _POSITIONAL_ATTR_PARAMS "
        "(if it is an attribute) or correct min_arity (if it is an operand)."
    )


def test_positional_attr_map_covers_every_declared_op():
    """Each entry must name a catalog op and declare at least one attribute.

    NOTE what this does NOT check, and why. An earlier version asserted that
    each declared name is a PARAMETER of the reference function. That is wrong:
    these are the **emitted Graph IR attribute names**, which deliberately
    differ from the Python parameter names — `permute(x, axes)` emits `perm`,
    `squeeze(x, axis)` emits `axes`, `flatten(x, start_axis, end_axis)` emits
    `start` + `end`. "Fixing" them to match the signature broke five frontend
    emission tests, which is how the mistake was caught.

    The map is therefore param-POSITION to attribute-NAME, not name to name.
    A gate that assumes otherwise enforces the wrong contract.
    """
    catalog = {spec.graph_name for spec in _SPECS}
    for graph_name, attrs in sorted(_POSITIONAL_ATTR_PARAMS.items()):
        assert graph_name in catalog, (
            f"{graph_name} declares positional attributes but is not a catalog op"
        )
        assert attrs, f"{graph_name} declares an empty attribute tuple"
        assert all(isinstance(a, str) and a for a in attrs), (
            f"{graph_name} has a malformed attribute name: {attrs}"
        )
