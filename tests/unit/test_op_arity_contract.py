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
  * **Keyword-only parameters are not all attributes.** This gate originally
    scoped itself to positional parameters because keyword-only ones had no
    declaration site. Building that vocabulary (W1.3) showed the caveat was
    hiding a defect rather than a mere gap: keyword-only params split into
    ATTRIBUTES (`softcap(x, *, cap)`) and real TENSOR OPERANDS
    (`attn_top_k_blocks(Q, K, V, *, scores)`), and the frontend was emitting the
    latter as string attributes holding an SSA name. Both halves are now
    declared -- `_KEYWORD_ATTR_PARAMS` and `_KEYWORD_OPERANDS` -- so the gate
    covers the full signature.

Both mistakes were made while building this, which is why the check states
them explicitly rather than just asserting a number.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from tessera.compiler.graph_ir import (
    _KEYWORD_ATTR_PARAMS,
    _KEYWORD_OPERANDS,
    _POSITIONAL_ATTR_PARAMS,
    _PRESENCE_FLAGGED_OPERANDS,
)
from tessera.compiler.op_catalog import _SPECS


def _required_positional_count(fn):
    """Required POSITIONAL parameters, or None if variadic."""
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


def _required_keyword_only(fn):
    """Required keyword-only parameter names, or None if the signature is
    unreadable. These split into attributes and tensor operands; the catalog
    declares which is which."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    return {
        p.name for p in signature.parameters.values()
        if p.kind is p.KEYWORD_ONLY and p.default is p.empty
    }


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
            # Keyword-only TENSOR operands count toward the semantic arity the
            # same as positional ones -- `attn_top_k_blocks`'s `scores` is an
            # operand no matter which side of the `*` it is written on. Missing
            # this is what let its min_arity sit at 3 for a 4-operand op.
            #
            # Only the REQUIRED ones, though: `moe`'s `scores` and `route` are
            # optional tensor operands (default None), so they raise max_arity
            # to 4 while min_arity stays 2. Counting every declared keyword
            # operand as required reported `moe` as drift when the catalog was
            # right -- the same operands-are-not-parameters error as above, one
            # level down.
            params = inspect.signature(fn).parameters
            required += sum(
                1 for name in _KEYWORD_OPERANDS.get(spec.graph_name, ())
                if name in params and params[name].default is inspect.Parameter.empty
            )
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


def test_every_required_keyword_only_param_is_classified():
    """Each required keyword-only param is declared an attribute OR an operand.

    Unlike `_POSITIONAL_ATTR_PARAMS` these ARE the reference's parameter names,
    because a keyword binds by name at the call site — so a rename in the
    signature must be reflected here or the emitter binds a stale key.

    The classification is the whole point. Both halves emit differently and
    getting it wrong is silent: an attribute misfiled as an operand inflates the
    arity, and an operand misfiled as an attribute becomes `{scores = "%s"}` —
    an SSA name in a string attribute, which erases the dataflow edge while
    still printing as valid-looking IR.
    """
    from tessera import ops

    unclassified = []
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
            required = _required_keyword_only(fn)
            if not required:
                continue
            declared = set(_KEYWORD_ATTR_PARAMS.get(spec.graph_name, ()))
            declared |= set(_KEYWORD_OPERANDS.get(spec.graph_name, ()))
            missing = sorted(required - declared)
            if missing:
                unclassified.append(f"{spec.public_name}: {missing}")

    assert not unclassified, (
        "required keyword-only parameters with no declaration:\n  "
        + "\n  ".join(unclassified)
        + "\nDeclare each in graph_ir._KEYWORD_ATTR_PARAMS (a scalar attribute) "
        "or _KEYWORD_OPERANDS (a tensor operand)."
    )


def test_keyword_declarations_name_real_parameters():
    """A declared keyword name must exist in the reference signature.

    This is the check `_POSITIONAL_ATTR_PARAMS` deliberately cannot have: there
    the map is position-to-emitted-name, so the names legitimately differ. Here
    they cannot differ, because the emitter matches on the keyword itself.
    """
    from tessera import ops

    stale = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for source, mapping in (("attr", _KEYWORD_ATTR_PARAMS),
                                ("operand", _KEYWORD_OPERANDS)):
            for graph_name, names in sorted(mapping.items()):
                spec = next((s for s in _SPECS if s.graph_name == graph_name), None)
                assert spec is not None, (
                    f"{graph_name} declares keyword {source}s but is not a catalog op"
                )
                fn = getattr(ops, spec.public_name, None)
                if fn is None:
                    continue
                params = inspect.signature(fn).parameters
                for name in names:
                    if name not in params:
                        stale.append(f"{spec.public_name}: {source} {name!r}")

    assert not stale, (
        "keyword declarations naming parameters the reference does not have:\n  "
        + "\n  ".join(stale)
    )


# ── the vocabulary gate graph_ir.py has always claimed to have ───────────────
#
# `graph_ir.py` told readers that undeclared keyword operands were "flagged by
# `test_keyword_operand_vocabulary` as vocabulary that needs declaring". No such
# test existed anywhere in the tree. The comment described a safety net that was
# never built, which is why `gated_deltanet`'s optional tensor operands stayed
# undeclared long enough to ship wrong numbers through the Apple GPU JIT path:
# undeclared, the emitter appends keyword operands **sorted by name**, so
# `gated_deltanet(q, k, v, gate=g, beta=b, decay=d)` emitted them in the order
# (beta, decay, gate) against an ABI of (gate, beta, decay).
#
# A promised gate is worse than an absent one: reviewers read the comment and
# assume the class is covered.


def _optional_operand_span(spec):
    """How many optional operands the catalog says this op accepts."""
    lo = getattr(spec, "min_arity", None)
    hi = getattr(spec, "max_arity", None)
    if lo is None or hi is None:
        return 0
    return max(0, int(hi) - int(lo))


#: Ops accepting several optional operands whose list is nonetheless decodable
#: without presence flags, with the reason. Two shapes qualify:
#:
#: * **homogeneous variadic** — every operand plays the same role, so there are
#:   no slots to confuse (`cat([a, b, c])`);
#: * **scalar-valued optionals** — the trailing parameters lower to attributes,
#:   which are already name-carrying.
#:
#: Everything else is tracked debt below, not an exemption.
_DECODABLE_WITHOUT_FLAGS = {
    "tessera.cat": "homogeneous variadic tensor list; no per-slot meaning",
    "tessera.stack": "homogeneous variadic tensor list; no per-slot meaning",
    "tessera.einsum": "homogeneous variadic operand list driven by the subscript",
    "tessera.arange": "start/stop/step lower to attributes, not operands",
}

#: Ops whose optional operands are NOT decodable from position alone and do not
#: yet emit presence flags. Each has the same latent defect the delta family
#: had: `layer_norm(x, beta=b)` puts `beta` at the index a positional reader
#: calls `gamma`.
#:
#: **This is a ratchet — it may shrink, never grow.** They are listed rather
#: than fixed here because each needs its consumers migrated with it, and doing
#: that blind, in the change that fixed a different op, is how a silent
#: mis-binding gets introduced rather than removed. The delta family is the
#: worked example to follow.
_UNDECODABLE_OPERAND_LISTS = {
    "tessera.adafactor",
    "tessera.adamw",
    "tessera.alibi",
    "tessera.conv2d_nhwc",
    "tessera.conv3d_ndhwc",
    "tessera.fused_epilogue",
    "tessera.group_norm",
    "tessera.instance_norm",
    "tessera.layer_norm",
    "tessera.mla_decode",
    "tessera.moe",
    "tessera.rl.ppo_policy_loss",
}


def test_keyword_operand_vocabulary():
    """Every op with several optional operands is classified.

    An op accepting more than one optional operand has an operand list that
    position alone cannot decode: given `[X, %a]` the lone optional sits at the
    same index whichever slot it fills. Such an op must either emit presence
    flags, be decodable for a stated reason, or be recorded as known debt.
    """
    unclassified = []
    for spec in _SPECS:
        name = spec.graph_name
        if _optional_operand_span(spec) <= 1:
            continue
        if (name in _PRESENCE_FLAGGED_OPERANDS
                or name in _DECODABLE_WITHOUT_FLAGS
                or name in _UNDECODABLE_OPERAND_LISTS):
            continue
        unclassified.append(name)

    assert not unclassified, (
        "ops accept several optional operands but nothing says how to decode "
        "the operand list:\n  " + "\n  ".join(sorted(set(unclassified)))
        + "\nDeclare presence flags in graph_ir._PRESENCE_FLAGGED_OPERANDS "
        "(and the order in _KEYWORD_OPERANDS), or record why the list is "
        "decodable without them."
    )


def test_the_undecodable_ratchet_only_shrinks():
    """The tracked-debt list may lose entries, never gain them.

    Its size is the number of ops that can still silently bind an optional
    operand to the wrong slot.
    """
    assert len(_UNDECODABLE_OPERAND_LISTS) <= 12, (
        "an op was added to _UNDECODABLE_OPERAND_LISTS. Fix it instead: "
        "declare its operand order and presence flags, and migrate its "
        "consumers in the same change."
    )
    resolved = _UNDECODABLE_OPERAND_LISTS & set(_PRESENCE_FLAGGED_OPERANDS)
    assert not resolved, (
        "these now emit presence flags and should be removed from the debt "
        f"list: {sorted(resolved)}"
    )


def test_presence_flags_agree_with_the_declared_operand_order():
    """A flagged op must declare the same names, in the same order.

    The two declarations are read by different halves of the emitter -- order
    when appending operands, flags when recording presence -- so a divergence
    would emit operands in one order and describe them in another.
    """
    for name, flagged in _PRESENCE_FLAGGED_OPERANDS.items():
        declared = _KEYWORD_OPERANDS.get(name)
        assert declared is not None, (
            f"{name} declares presence flags but no operand order; the flags "
            "would describe positions nothing pinned"
        )
        assert tuple(flagged) == tuple(declared), (
            f"{name} flags {flagged} but declares operand order {declared}"
        )


def test_flagged_ops_name_real_optional_parameters():
    """Each flagged name must be an actual optional parameter of the op."""
    from tessera import ops

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for graph_name, flagged in _PRESENCE_FLAGGED_OPERANDS.items():
            spec = next((s for s in _SPECS if s.graph_name == graph_name), None)
            assert spec is not None, f"{graph_name} is not a catalog op"
            fn = getattr(ops, spec.public_name, None)
            if fn is None:
                continue
            params = inspect.signature(fn).parameters
            for entry in flagged:
                assert entry in params, (
                    f"{graph_name} flags {entry!r}, absent from the reference "
                    f"signature {tuple(params)}"
                )
                assert params[entry].default is not inspect.Parameter.empty, (
                    f"{graph_name} flags {entry!r}, but it is required -- a "
                    "presence flag on a mandatory operand is always True"
                )
