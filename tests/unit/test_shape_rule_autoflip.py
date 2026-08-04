"""W1.2 — the `shape_rule` axis is DERIVED from `op_catalog`, not hand-declared.

`op_catalog` predicted this mechanism in its own source, and then nobody wired
it:

    "`primitive_coverage.shape_rule` can auto-flip from real declarations the
     same way it already does from `_VJPS` / `_JVPS`."

Until it was wired, the dashboard promoted `shape_rule` from `partial` to
`complete` off the **lowering kind** and never consulted the catalog. That is
the same bug `shape_rule_for` had already fixed one layer down, and its comment
says exactly why it matters:

    "A deliberate non-declaration outranks any lowering-kind default. Without
     this, an op could be listed as 'examined, deliberately undeclared' while
     `shape_rule_for` still handed back its kind's default -- and the gates
     would enforce a rule the catalog had explicitly withdrawn."

Measured before this landed: all six ops the catalog had **explicitly withdrawn**
a shape rule for reported `complete` — a closed contract for a rule that does
not exist. Two registries asserting the same fact independently is Decision
#24/#29 in miniature, and the drift always resolves toward the more flattering
answer.
"""

from __future__ import annotations

import pytest

from tessera.compiler.op_catalog import graph_name_for, shape_rule_for
from tessera.compiler.primitive_coverage import (
    _catalog_shape_rule_status,
    all_primitive_coverages,
)

#: The catalog's six deliberate non-declarations. Four are the callable-operand
#: family (their operand 0 is a Python function, so there is no operand type to
#: read); two are host-side training entry points.
_WITHDRAWN = {
    "dz", "dbar", "conformal_jacobian", "check_cauchy_riemann",
    "training.loss_adamw", "training.loss_sgd",
}


@pytest.fixture(scope="module")
def registry():
    return all_primitive_coverages()


def test_no_op_claims_a_shape_rule_the_catalog_withdrew(registry):
    """The regression this item exists for.

    An op the catalog deliberately does not declare a rule for must not report
    a closed contract. Before the auto-flip all six did.
    """
    offenders = []
    for name, entry in registry.items():
        graph_name = graph_name_for(name)
        if graph_name is None:
            continue
        if (shape_rule_for(graph_name) == "unclassified"
                and entry.contract_status.get("shape_rule") == "complete"):
            offenders.append(name)
    assert not offenders, (
        f"these ops report shape_rule=complete while op_catalog declares no "
        f"rule for them: {sorted(offenders)}"
    )


def test_the_withdrawn_six_are_still_the_withdrawn_six():
    """Pins the set, so the gate above cannot pass by the set shrinking to zero.

    A ratchet that only checks a property of a set is satisfied by an empty
    set. This one names the members.
    """
    actual = {
        name for name in _WITHDRAWN
        if (g := graph_name_for(name)) and shape_rule_for(g) == "unclassified"
    }
    assert actual == _WITHDRAWN, (
        "the catalog's deliberate non-declarations changed; update _WITHDRAWN "
        "and confirm the new set is intentional"
    )


def test_every_catalog_owned_op_matches_the_derivation(registry):
    """The axis is derived, full stop — no per-name override may contradict it.

    39 single-line `"shape_rule": "complete"` entries survive in the override
    tables. They are inert (the derivation runs last) and every one was measured
    to AGREE with the catalog. This gate is what keeps that true: add a
    contradicting override and the build fails instead of the dashboard quietly
    reporting the more flattering of two answers.
    """
    mismatches = []
    for name, entry in registry.items():
        derived = _catalog_shape_rule_status(name)
        if derived is None:
            continue
        if entry.contract_status.get("shape_rule") != derived:
            mismatches.append(
                (name, entry.contract_status.get("shape_rule"), derived)
            )
    assert not mismatches, (
        f"dashboard disagrees with op_catalog on shape_rule: {mismatches[:10]}"
    )


def test_ops_the_catalog_does_not_own_are_left_alone():
    """The derivation must not reach past its evidence.

    ~169 primitives have no Graph IR name — the Python-reference and host-API
    surface. The catalog has no rule to offer for them, so the helper returns
    None and they keep whatever the category tables say. Asserting `complete`
    OR demoting them here would both be inventing an answer; the scope of this
    item is the ops the catalog actually owns.
    """
    assert _catalog_shape_rule_status("avg_pool") is None
    assert _catalog_shape_rule_status("autocast") is None
    # ...and an op it does own resolves.
    assert _catalog_shape_rule_status("matmul") == "complete"


def test_derivation_is_a_pure_function_of_the_catalog():
    """Same catalog answer in, same axis status out — the `_VJPS` property.

    Stated because the value of an auto-flip is precisely that it cannot be
    edited independently of its source.
    """
    for name in ("matmul", "flash_attn", "dz"):
        graph_name = graph_name_for(name)
        if graph_name is None:
            continue
        expected = ("partial" if shape_rule_for(graph_name) == "unclassified"
                    else "complete")
        assert _catalog_shape_rule_status(name) == expected
