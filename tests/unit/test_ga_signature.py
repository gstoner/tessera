"""GA1 acceptance: Clifford algebra signature object.

Sprint: GA1 (algebra signature as a first-class object).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md
Scope lock: docs/audit/domain/DOMAIN_AUDIT.md

Covers the GA1 acceptance criteria:
  - Cl(3,0).dim == 8, Cl(1,3).dim == 16
  - Product table cached per signature
  - Hashable + value-equal across calls
  - Grade tuple matches (0, 1, ..., n)
  - blade("e12").grade == 2 — canonical name parsing
  - Per-generator squares (e_i * e_i = +1 / -1 / 0) match signature
  - Wedge anti-commutation: e1*e2 == -(e2*e1)
  - Cayley-table associativity (sampled)
  - Repr / pickle round-trip
  - Construction validation: v1 allow-list + non-negative guard
"""

from __future__ import annotations

import pickle

import pytest

from tessera.ga import Basis, Cl, TesseraAlgebraError, V1_ALLOWED_SIGNATURES


# ---------------------------------------------------------------------------
# Dimensions, grades, basis enumeration
# ---------------------------------------------------------------------------

def test_cl30_has_8_basis_elements() -> None:
    a = Cl(3, 0)
    assert a.n == 3
    assert a.dim == 8
    assert a.grades == (0, 1, 2, 3)
    assert len(a.blades()) == 8


def test_cl13_has_16_basis_elements() -> None:
    a = Cl(1, 3)
    assert a.n == 4
    assert a.dim == 16
    assert a.grades == (0, 1, 2, 3, 4)
    assert len(a.blades()) == 16


def test_cl30_blade_names_match_textbook() -> None:
    a = Cl(3, 0)
    names = [b.name for b in a.blades()]
    assert names == ["1", "e1", "e2", "e12", "e3", "e13", "e23", "e123"]


def test_cl30_blade_lookup() -> None:
    a = Cl(3, 0)
    assert a.blade("1").mask == 0
    assert a.blade("e1").mask == 0b001
    assert a.blade("e2").mask == 0b010
    assert a.blade("e3").mask == 0b100
    assert a.blade("e12").mask == 0b011
    assert a.blade("e13").mask == 0b101
    assert a.blade("e23").mask == 0b110
    assert a.blade("e123").mask == 0b111
    assert a.blade("e12").grade == 2
    assert a.blade("e123").grade == 3


def test_scalar_and_pseudoscalar() -> None:
    a = Cl(3, 0)
    assert a.scalar.mask == 0
    assert a.scalar.grade == 0
    assert a.pseudoscalar.mask == 0b111
    assert a.pseudoscalar.grade == 3
    assert a.pseudoscalar.name == "e123"

    b = Cl(1, 3)
    assert b.pseudoscalar.grade == 4
    assert b.pseudoscalar.name == "e1234"


def test_blades_of_grade_cl30() -> None:
    a = Cl(3, 0)
    assert {b.name for b in a.blades_of_grade(0)} == {"1"}
    assert {b.name for b in a.blades_of_grade(1)} == {"e1", "e2", "e3"}
    assert {b.name for b in a.blades_of_grade(2)} == {"e12", "e13", "e23"}
    assert {b.name for b in a.blades_of_grade(3)} == {"e123"}


# ---------------------------------------------------------------------------
# Product table — per-signature squaring rules
# ---------------------------------------------------------------------------

def test_cl30_generator_squares_to_plus_one() -> None:
    a = Cl(3, 0)
    table = a.product_table()
    for blade in a.blades_of_grade(1):
        mask, sign = table[blade.mask][blade.mask]
        assert mask == 0  # scalar
        assert sign == 1, f"{blade.name}**2 expected +1, got {sign}"


def test_cl13_first_generator_squares_to_plus_one() -> None:
    a = Cl(1, 3)
    table = a.product_table()
    e1 = a.blade("e1")
    mask, sign = table[e1.mask][e1.mask]
    assert (mask, sign) == (0, 1)


def test_cl13_remaining_generators_square_to_minus_one() -> None:
    a = Cl(1, 3)
    table = a.product_table()
    for name in ("e2", "e3", "e4"):
        b = a.blade(name)
        mask, sign = table[b.mask][b.mask]
        assert mask == 0
        assert sign == -1, f"{name}**2 expected -1, got {sign}"


def test_basis_blade_anticommutes() -> None:
    a = Cl(3, 0)
    table = a.product_table()
    e1, e2 = a.blade("e1").mask, a.blade("e2").mask
    e12 = a.blade("e12").mask
    assert table[e1][e2] == (e12, 1)   # e1 * e2 = e12
    assert table[e2][e1] == (e12, -1)  # e2 * e1 = -e12


def test_pseudoscalar_squares_match_signature() -> None:
    # In Cl(3,0): I^2 = -1.   In Cl(1,3): I^2 = -1 (since I = e1234, I^2 = -1).
    a = Cl(3, 0)
    table_a = a.product_table()
    I_a = a.pseudoscalar.mask
    assert table_a[I_a][I_a] == (0, -1)

    b = Cl(1, 3)
    table_b = b.product_table()
    I_b = b.pseudoscalar.mask
    assert table_b[I_b][I_b] == (0, -1)


# ---------------------------------------------------------------------------
# Cayley-table associativity (sampled)
# ---------------------------------------------------------------------------

def _signed_product(table, a: int, b: int) -> tuple[int, int]:
    return table[a][b]


def _compose_signed(table, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    mask_a, sign_a = a
    mask_b, sign_b = b
    res_mask, res_sign = table[mask_a][mask_b]
    return res_mask, sign_a * sign_b * res_sign


@pytest.mark.parametrize("signature", [(3, 0, 0), (1, 3, 0)])
def test_cayley_table_is_associative(signature) -> None:
    a = Cl(*signature)
    table = a.product_table()
    blades = [b.mask for b in a.blades()]
    # Sampled triples: every (a, b, c) where each is a basis blade —
    # full sweep is 16^3 = 4096 in Cl(1,3), well within budget.
    for ma in blades:
        for mb in blades:
            ab = _signed_product(table, ma, mb)
            for mc in blades:
                # (a*b)*c
                lhs = _compose_signed(table, ab, (mc, 1))
                # a*(b*c)
                bc = _signed_product(table, mb, mc)
                rhs = _compose_signed(table, (ma, 1), bc)
                assert lhs == rhs, (
                    f"associativity failed in Cl{signature[:2]} at "
                    f"({ma:b}, {mb:b}, {mc:b}): {lhs} vs {rhs}"
                )


# ---------------------------------------------------------------------------
# Caching, equality, hashing, pickle
# ---------------------------------------------------------------------------

def test_signatures_are_value_equal_and_hashable() -> None:
    assert Cl(3, 0) == Cl(3, 0)
    assert hash(Cl(3, 0)) == hash(Cl(3, 0))
    assert Cl(3, 0) != Cl(1, 3)
    # Hashable: usable as dict key.
    {Cl(3, 0): "A", Cl(1, 3): "B"}


def test_product_table_is_cached_per_signature() -> None:
    """Second `product_table()` call returns the same tuple instance."""
    a = Cl(3, 0)
    b = Cl(3, 0)
    assert a.product_table() is b.product_table()
    # Different signatures get different tables.
    assert Cl(3, 0).product_table() is not Cl(1, 3).product_table()


def test_pickle_round_trip() -> None:
    a = Cl(3, 0)
    b = pickle.loads(pickle.dumps(a))
    assert a == b
    assert b.signature == (3, 0, 0)
    assert b.product_table() is Cl(3, 0).product_table()  # cache survives


def test_repr_drops_zero_r() -> None:
    assert repr(Cl(3, 0)) == "Cl(3, 0)"
    assert repr(Cl(1, 3)) == "Cl(1, 3)"


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_v1_allow_list_blocks_unsupported_signatures() -> None:
    # Cl(4,1) — conformal — should be rejected per v1 lock.
    with pytest.raises(TesseraAlgebraError, match="v1 allow-list"):
        Cl(4, 1)
    # Cl(2,0) — would be quaternion-like — also rejected for v1.
    with pytest.raises(TesseraAlgebraError, match="v1 allow-list"):
        Cl(2, 0)
    # Cl(3,0,1) — degenerate (r > 0) — rejected for v1.
    with pytest.raises(TesseraAlgebraError, match="v1 allow-list"):
        Cl(3, 0, 1)


def test_v1_allow_list_contains_exactly_two_signatures() -> None:
    assert V1_ALLOWED_SIGNATURES == frozenset({(3, 0, 0), (1, 3, 0)})


def test_negative_parameters_rejected() -> None:
    with pytest.raises(TesseraAlgebraError, match="non-negative"):
        Cl(-1, 0)
    with pytest.raises(TesseraAlgebraError, match="non-negative"):
        Cl(3, -1)


def test_non_int_parameters_rejected() -> None:
    with pytest.raises(TesseraAlgebraError, match="must be int"):
        Cl(3.0, 0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Blade name parser — strictness
# ---------------------------------------------------------------------------

def test_blade_lookup_rejects_non_ascending_names() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="canonical"):
        a.blade("e21")
    with pytest.raises(TesseraAlgebraError, match="canonical"):
        a.blade("e321")


def test_blade_lookup_rejects_out_of_range() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="e4"):
        a.blade("e4")


def test_blade_lookup_rejects_repeated_generators() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="repeats"):
        a.blade("e11")


def test_blade_lookup_rejects_malformed_names() -> None:
    a = Cl(3, 0)
    with pytest.raises(TesseraAlgebraError, match="Invalid blade name"):
        a.blade("foo")
    with pytest.raises(TesseraAlgebraError, match="Invalid blade name"):
        a.blade("e")
    with pytest.raises(TesseraAlgebraError, match="Invalid blade name"):
        a.blade("eA")


# ---------------------------------------------------------------------------
# Version stamp
# ---------------------------------------------------------------------------

def test_ga_version_at_least_ga1() -> None:
    """The signature object exists from GA1 onwards; allow later sprint bumps."""
    from tessera import ga

    # Version stamp is 0.0.0-ga<N>; any N >= 1 is acceptable once the
    # signature object is live.
    assert ga.__version__.startswith("0.0.0-ga")
    sprint_str = ga.__version__.split("-ga", 1)[1]
    assert int(sprint_str) >= 1
