"""Hold the required-contract count from falling as the Target IR expands.

Decision #19's amendment invites Apple (~12 ops) and x86 (~8) to grow their
dialects, and expansion is exactly when contract carriage regresses: the
cheapest way to add an op is to derive it from the dialect's op base class and
inherit its `OptionalAttr` bag, which produces surface that *looks*
contract-carrying and enforces nothing.
"""

from __future__ import annotations

from tessera.compiler import target_ir_membership as membership


def test_base_class_attributes_are_attributed_to_derived_ops():
    """The measurement is worthless if inheritance is invisible.

    `class TesseraNVIDIA_Op<string mnemonic, list<Trait> traits = []> :` nests
    angle brackets, so a `<[^>]*>` pattern stops inside `list<Trait>`, finds no
    base classes, and reports every inheriting op as carrying no contract. That
    is a silent 0%, not an error — which is why it is asserted rather than
    assumed.
    """
    rows = {(r.backend, r.mnemonic): r for r in membership.collect()}
    mfma = rows[("rocm", "mfma")]
    assert mfma.required or mfma.optional, (
        "rocm.mfma inherits its contract from TesseraROCM_Op; seeing none means "
        "base-class attribution broke"
    )


def test_enum_and_type_definitions_are_not_counted_as_ops():
    names = {r.mnemonic for r in membership.collect()}
    assert "abs_i32" not in names and "uniform_core" not in names


def test_required_contract_count_does_not_regress():
    """Ratchet. Raise the floor when it improves; never lower it to pass."""
    per = membership.summary()
    requires = sum(b["requires"] for b in per.values())
    assert requires >= 38, (
        f"{requires} ops require their contract, was 38 — an op was added or "
        "changed that declares its contract as optional. Decision #19's layer "
        "exists for contract carriage; an OptionalAttr contract fails open."
    )


def test_every_op_lands_in_exactly_one_verdict():
    for row in membership.collect():
        assert row.verdict in {"requires", "optional-only", "no-contract"}
        if row.verdict == "requires":
            assert row.required
        elif row.verdict == "optional-only":
            assert not row.required and row.optional
        else:
            assert not row.required and not row.optional


def test_apple_is_reported_as_the_weakest_backend():
    """The finding that shapes the Apple work, asserted so a fix is visible.

    Apple requires no contract at all — the same gap as its missing machine
    primitives, seen from the contract side: a dialect of dispatch containers
    has nothing to enforce. When this fails because Apple grew a required
    contract, replace the assertion with the evidence rather than deleting it.
    """
    per = membership.summary()
    assert per["apple"]["requires"] == 0
    assert per["rocm"]["requires"] > per["nvidia"]["requires"]
