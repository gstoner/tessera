"""Executable mathematical contract for the CuTe IR review (NVIDIA/cutlass#3426).

Verifies every worked example in the CuTe IR layout-algebra documentation by
independent evaluation -- not by trusting the prose, and not by linking cutegen.
See docs/audit/compiler/CUTE_IR_ASSESSMENT.md Sec. 1 for the findings these
checks produce.

A layout is a pair (shape, stride) of equal-length tuples, evaluated at a 1-D
coordinate `i` by decomposing `i` column-major over `shape` and dotting with
`stride`. Mode *nesting* changes only how the modes are bracketed, never the
function -- so a flattened encoding is faithful, and every claim below reduces
to comparing integer functions over a finite domain. That is the whole reason
this substrate is worth importing: it is exhaustively checkable with no
hardware, which is Decision #19's discipline applied to index arithmetic.

Three of these tests assert that a DOCUMENTED CLAIM IS FALSE. They are negative
fixtures on purpose (Decision #10a): the CuTe IR tutorial states three things
its own examples contradict, and a Tessera port that reads the prose as the
contract would build a wrong verifier. If a future CuTe release fixes the
prose, these three tests are the ones that must be revisited -- they are not
regressions in our code.

Dependency-free by construction (no numpy, no torch): pure integer arithmetic.
"""

from __future__ import annotations

from math import prod

import pytest

Layout = tuple[tuple[int, ...], tuple[int, ...]]


def ev(layout: Layout, i: int) -> int:
    """Evaluate a layout at 1-D coordinate `i` (column-major decomposition)."""
    shape, stride = layout
    idx, rest = 0, i
    for s, d in zip(shape, stride):
        idx += (rest % s) * d
        rest //= s
    return idx


def img(layout: Layout) -> list[int]:
    """The layout's image over its full domain [0, size)."""
    return [ev(layout, i) for i in range(prod(layout[0]))]


def leaves(layout: Layout) -> list[tuple[int, int]]:
    """Sorted (extent, stride) leaf pairs -- the grouping-invariant signature."""
    return sorted(zip(*layout))


# ---------------------------------------------------------------- coalesce


def test_coalesce_preserves_the_function():
    """(2,(1,6)):(1,(6,2)) -> 12:1 -- fewer modes, identical map."""
    assert img(((2, 1, 6), (1, 6, 2))) == img(((12,), (1,)))


def test_coalesce_by_mode_only_fires_inside_each_mode():
    """(3,(4,5)):(8,(1,4)) with profile (1,1) -> (3,20):(8,1), still rank 2."""
    assert img(((3, 4, 5), (8, 1, 4))) == img(((3, 20), (8, 1)))


# ------------------------------------------------------------- composition


def assert_composition(lhs: Layout, rhs: Layout, result: Layout) -> None:
    """R(c) == lhs(rhs(c)) for every c in rhs's domain."""
    assert prod(rhs[0]) == prod(result[0])
    for j in range(prod(rhs[0])):
        assert ev(lhs, ev(rhs, j)) == ev(result, j), f"mismatch at {j}"


def test_composition_splits_a_mode_across_an_lhs_boundary():
    """(6,2):(8,2) o (4,3):(3,1) -> ((2,2),3):((24,2),8).

    rhs mode 0 (extent 4) straddles lhs mode 0 (extent 6), so it splits into
    (2,2). This is the case a naive stride-multiply gets wrong.
    """
    assert_composition(((6, 2), (8, 2)), ((4, 3), (3, 1)), ((2, 2, 3), (24, 2, 8)))


def test_composition_with_a_tile_is_mode_wise_not_functional():
    """(6,4):(4,1) o [(2):(1);(2):(1)] -> ((2),(2)):((4),(1)).

    A tile is a per-mode tiler: slot i drives input mode i independently. It is
    NOT one flat layout composed as a function -- that reading gives a different
    answer, and is the first thing a reimplementation gets wrong.
    """
    src, res = ((6, 4), (4, 1)), ((2, 2), (4, 1))
    for c0 in range(2):
        for c1 in range(2):
            assert ev(src, c0 + 6 * c1) == ev(res, c0 + 2 * c1)


def test_DOC_DEFECT_composition_with_shape_prose_contradicts_its_own_example():
    """core_ops.rst calls shape-composition `A o make_layout(shape)`. It is not.

    The correct reading -- and the one CuteOps.td states -- is mode-wise
    truncation: mode i of A restricted to shape[i]. The tutorial's stated
    equivalence yields a different function than the tutorial's own example.
    """
    a = ((4, 8), (1, 4))
    via_prose = [ev(a, ev(((2, 4), (1, 2)), j)) for j in range(8)]
    documented_result = img(((2, 4), (1, 4)))

    assert via_prose == [0, 1, 2, 3, 4, 5, 6, 7]
    assert documented_result == [0, 1, 4, 5, 8, 9, 12, 13]
    assert via_prose != documented_result, "prose and example agree -- doc fixed?"

    # Mode-wise truncation is what actually reproduces the documented result.
    mode_wise = ((2, 4), (1, 4))  # mode0 4:1 -> 2:1, mode1 8:4 -> 4:4
    assert img(mode_wise) == documented_result


# -------------------------------------------------------------- complement


A_COMPLEMENT_SRC: Layout = ((2, 2), (4, 1))  # image {0,1,4,5}, cosize 6


def test_complement_with_cotarget_tiles_the_codomain_exactly():
    """complement((2,2):(4,1), 24) == (2,3):(2,8) -- A (+) A* bijects [0,24)."""
    sums = sorted(
        a + c for a in img(A_COMPLEMENT_SRC) for c in img(((2, 3), (2, 8)))
    )
    assert sums == list(range(24))


def test_DOC_DEFECT_complement_without_cotarget_overshoots_its_stated_guarantee():
    """The no-cotarget form does not "cover exactly [0,M) with no overlap".

    M defaults to cosize(A) = 6, but the documented answer (2,1):(2,8) yields a
    bijection onto [0,8) -- a strict superset. It does reach the codomain holes
    {2,3}, so the form is usable; the *guarantee as written* is false. Traces to
    shape_div saturating 6/8 to 1 rather than 0.
    """
    cover = sorted(
        a + c for a in img(A_COMPLEMENT_SRC) for c in img(((2, 1), (2, 8)))
    )
    assert cover != list(range(6)), "guarantee now holds -- doc fixed?"
    assert cover == list(range(8))          # bijection onto its own span
    assert {2, 3} <= set(cover)             # the holes are reached


# ---------------------------------------------------------------- inverses


@pytest.mark.parametrize(
    "src, inverse",
    [
        (((2, 4, 6), (4, 1, 8)), ((4, 2, 6), (2, 1, 8))),
        (((2, 4), (4, 1)), ((4, 2), (2, 1))),
    ],
)
def test_right_inverse_identity(src: Layout, inverse: Layout):
    """A(A_R^-1(j)) == j over the inverse's domain."""
    for j in range(prod(inverse[0])):
        assert ev(src, ev(inverse, j)) == j


def test_worked_example_rows_are_as_printed():
    """The tutorial prints both walks explicitly; check them digit for digit."""
    assert img(((2, 4), (4, 1))) == [0, 4, 1, 5, 2, 6, 3, 7]
    assert img(((4, 2), (2, 1))) == [0, 2, 4, 6, 1, 3, 5, 7]


def test_left_inverse_identity_for_a_bijective_layout():
    """A_L^-1(A(c)) == c -- equal to the right inverse when A is a bijection."""
    src, inv = ((2, 4, 6), (4, 1, 8)), ((4, 2, 6), (2, 1, 8))
    assert sorted(img(src)) == list(range(48)), "precondition: A is bijective"
    for c in range(48):
        assert ev(inv, ev(src, c)) == c


# ------------------------------------------------------------------ recast


@pytest.mark.parametrize(
    "new_bits, old_bits, src, result",
    [
        (32, 8, ((32, 4), (1, 32)), ((8, 4), (1, 8))),     # upcast x4
        (8, 32, ((8, 4), (1, 8)), ((32, 4), (1, 32))),     # downcast x4
        (32, 32, ((8, 4), (1, 8)), ((8, 4), (1, 8))),      # identity
        (4, 6, ((8, 4), (1, 8)), ((12, 4), (1, 12))),      # general, gcd split
    ],
)
def test_recast_layout_preserves_total_bit_extent(new_bits, old_bits, src, result):
    """Recasting changes the element unit, never the number of bits addressed."""
    assert prod(src[0]) * old_bits == prod(result[0]) * new_bits
    # the stride-1 mode's run length, in bits, is also preserved
    assert src[0][0] * old_bits == result[0][0] * new_bits


def test_recast_general_case_is_upcast_then_downcast():
    """new=4 old=6: G=gcd=2, new'=2, old'=3 -- each step conserves bits."""
    src = ((8, 4), (1, 8))                    # in 6-bit units
    after_upcast = ((4, 4), (1, 4))           # 6-bit -> 12-bit units (x new'=2)
    after_downcast = ((12, 4), (1, 12))       # 12-bit -> 4-bit units (x old'=3)

    assert src[0][0] * 6 == after_upcast[0][0] * 12
    assert src[1][1] * 6 == after_upcast[1][1] * 12
    assert after_upcast[0][0] * 12 == after_downcast[0][0] * 4
    assert after_upcast[1][1] * 12 == after_downcast[1][1] * 4


# ---------------------------------------------------------------- products


PRODUCT_VARIANTS: dict[str, Layout] = {
    # input (3,4):(4,1), tiler (2,5):(1,2)
    "logical_product": ((3, 4, 2, 5), (4, 1, 12, 24)),
    "zipped_product": ((3, 4, 2, 5), (4, 1, 12, 24)),
    "tiled_product": ((3, 4, 2, 5), (4, 1, 12, 24)),
    "flat_product": ((3, 4, 2, 5), (4, 1, 12, 24)),
    "blocked_product": ((3, 2, 4, 5), (4, 12, 1, 24)),
    "raked_product": ((2, 3, 5, 4), (12, 4, 24, 1)),
}


def test_all_six_product_variants_are_one_leaf_multiset_regrouped():
    """The six variants differ only in bracketing -- same (extent,stride) set.

    This is the scoping result: the product family is one construction plus
    five regroupings, not six algorithms.
    """
    base = leaves(PRODUCT_VARIANTS["logical_product"])
    for name, variant in PRODUCT_VARIANTS.items():
        assert leaves(variant) == base, name


def test_logical_product_places_disjoint_copies():
    """Each copy of the block lands in its own slice of the codomain."""
    assert sorted(img(((2, 2, 2, 3), (4, 1, 2, 8)))) == list(range(24))
    assert sorted(img(((2, 2, 3, 2), (4, 1, 8, 2)))) == list(range(24))
    assert len(set(img(PRODUCT_VARIANTS["logical_product"]))) == 120


def test_DOC_DEFECT_product_table_gives_logical_product_the_wrong_grouping():
    """The table's logical_product row is verbatim its blocked_product row.

    The worked examples show the two are different groupings; the example
    matches the standard definition (A, complement(A,.) o B).
    """
    assert (
        PRODUCT_VARIANTS["logical_product"] != PRODUCT_VARIANTS["blocked_product"]
    ), "groupings now agree -- doc fixed?"
    assert PRODUCT_VARIANTS["logical_product"][0] == (3, 4, 2, 5)
    assert PRODUCT_VARIANTS["blocked_product"][0] == (3, 2, 4, 5)


def test_tile_to_shape_covers_the_target_bijectively():
    """block (2,2):(1,2) over target (8,8) -> ((2,4),(2,4)):((1,4),(2,16))."""
    assert sorted(img(((2, 4, 2, 4), (1, 4, 2, 16)))) == list(range(64))


# ----------------------------------------------------------------- divides


DIVIDE_SRC: Layout = ((6, 8), (8, 1))

DIVIDE_VARIANTS: dict[str, Layout] = {
    "logical_divide": ((3, 2, 4, 2), (8, 24, 1, 4)),
    "zipped_divide": ((3, 4, 2, 2), (8, 1, 24, 4)),
    "tiled_divide": ((3, 4, 2, 2), (8, 1, 24, 4)),
    "flat_divide": ((3, 4, 2, 2), (8, 1, 24, 4)),
}


def test_all_four_divide_variants_are_one_leaf_multiset_regrouped():
    base = leaves(DIVIDE_VARIANTS["logical_divide"])
    for name, variant in DIVIDE_VARIANTS.items():
        assert leaves(variant) == base, name


def test_divides_preserve_the_input_image():
    """A divide regroups the source layout; it never changes what it addresses."""
    expected = sorted(img(DIVIDE_SRC))
    for name, variant in DIVIDE_VARIANTS.items():
        assert sorted(img(variant)) == expected, name


def test_logical_divide_by_a_non_identity_layout_tiler_splits_a_mode():
    """(6,8):(8,1) / (3,4):(1,3) -> ((3,(2,2)),4):((8,(24,1)),2).

    The subtle one, and the strongest evidence the algebra is real. With
    complement((3,4):(1,3), 48) = 4:12, the inner map ((3,4),4):((1,3),12) is
    the identity on [0,48) -- yet the result is NOT the source regrouped,
    because A(j) = 8*l0 + 24*(l1 mod 2) + (l1 div 2) + 2*l2 is genuinely
    non-affine in the l1 mode. The nested (2,2) is that split, represented
    exactly rather than approximated.
    """
    inner = ((3, 4, 4), (1, 3, 12))
    assert img(inner) == list(range(48)), "inner map is the identity"

    result = ((3, 2, 2, 4), (8, 24, 1, 2))
    for j in range(48):
        assert ev(DIVIDE_SRC, ev(inner, j)) == ev(result, j), f"mismatch at {j}"


# ------------------------------------------------------- composed layouts


def test_composed_layout_applies_offset_between_the_two_maps():
    """(6,2):(1,3) o (2,1) o (2,3):(1@1,2@0) -- basis strides then offset then A.

    B's scaled-basis strides make B(c) a TUPLE, so the offset is added
    per-component before A indexes it. Getting this wrong (adding a scalar)
    silently produces plausible indices.
    """
    outer = ((6, 2), (1, 3))
    got = []
    for j in range(6):
        c0, c1 = j % 2, j // 2
        b0, b1 = 2 * c1, c0                      # B(c) -> (basis0, basis1)
        got.append((b0 + 2) * 1 + (b1 + 1) * 3)  # A(offset + B(c))
    assert got == [5, 8, 7, 10, 9, 12]


def test_composed_layout_example_evaluates_A_outside_its_own_shape():
    """A = (6,2):(1,3) is indexed at n0 = 6, one past its extent.

    Legal in CuTe -- a layout is an affine function defined beyond its shape --
    but it means the composed layout's image can exceed A's own cosize (9 here,
    reached 12). Any Tessera bounds verifier must model this deliberately
    rather than assume containment.
    """
    outer_shape, outer_cosize = (6, 2), 5 * 1 + 1 * 3 + 1
    assert outer_cosize == 9

    max_n0 = max((2 * (j // 2)) + 2 for j in range(6))
    assert max_n0 == 6 >= outer_shape[0], "n0 reaches A's extent, not below it"

    max_image = max([5, 8, 7, 10, 9, 12])
    assert max_image == 12 > outer_cosize, "image exceeds A's own cosize"
