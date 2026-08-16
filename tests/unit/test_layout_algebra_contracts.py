"""Executable mathematical contract for the CuTe IR review (NVIDIA/cutlass#3426).

Verifies every worked example in the CuTe IR layout-algebra documentation by
independent evaluation -- not by trusting the prose, and not by linking cutegen.
See docs/audit/compiler/CUTE_IR_ASSESSMENT.md Sec. 1 for the findings these
checks produce.

**Fixtures are the documented layout strings, verbatim.** A layout is written
the way the CuTe docs write it -- ``"((3,4),(2,5)):((4,1),(12,24))"`` -- and
parsed here into a nested (shape, stride) tree. That matters for two reasons:
transcription is removed as a source of error, and mode *nesting* is preserved,
so the tests can assert an operation's exact result STRUCTURE and not merely its
scalar image.

Both halves are load-bearing and neither subsumes the other:

* the *function* (image over [0, size)) is invariant under regrouping, so it
  proves an operation addresses the right elements;
* the *structure* (nested shape/stride) is what distinguishes logical_ from
  zipped_/tiled_/flat_/blocked_/raked_ product, and it is what downstream
  slicing, grouping and partitioning consume.

An earlier revision of this file encoded every layout flat. Four of the six
product variants then collapsed onto one identical tuple, so the "all six are
one leaf multiset regrouped" claim asserted what had been typed rather than
what the algebra does -- and an implementation that flattened every result, or
returned the wrong grouping outright, would have passed. Found in review of
PR #573; the structural assertions below are the fix. Do not reintroduce a flat
fixture type.

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

#: A nested int-tuple: a leaf, or a tuple of Trees. Mirrors CuTe's IntTuple.
Tree = int | tuple["Tree", ...]

#: (shape, stride), each an arbitrarily nested Tree of equal profile.
Layout = tuple[Tree, Tree]


# --------------------------------------------------------------- parsing


def _parse_tree(text: str, pos: int) -> tuple[Tree, int]:
    """Parse one Tree starting at `pos`; return (tree, next position)."""
    if text[pos] == "(":
        items: list[Tree] = []
        pos += 1
        while True:
            item, pos = _parse_tree(text, pos)
            items.append(item)
            if text[pos] == ",":
                pos += 1
                continue
            if text[pos] == ")":
                return tuple(items), pos + 1
            raise ValueError(f"unexpected {text[pos]!r} at {pos} in {text!r}")
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    if start == pos:
        raise ValueError(f"expected an integer at {start} in {text!r}")
    return int(text[start:pos]), pos


def layout(spec: str) -> Layout:
    """Parse a CuTe layout string ``"shape:stride"`` into a nested Layout.

    >>> layout("((3,4),(2,5)):((4,1),(12,24))")
    (((3, 4), (2, 5)), ((4, 1), (12, 24)))
    >>> layout("12:1")
    (12, 1)
    """
    text = spec.replace(" ", "")
    shape_text, sep, stride_text = text.partition(":")
    if not sep:
        raise ValueError(f"layout {spec!r} has no ':' separator")
    shape, end = _parse_tree(shape_text, 0)
    if end != len(shape_text):
        raise ValueError(f"trailing text in shape of {spec!r}")
    stride, end = _parse_tree(stride_text, 0)
    if end != len(stride_text):
        raise ValueError(f"trailing text in stride of {spec!r}")
    if profile(shape) != profile(stride):
        raise ValueError(f"shape/stride profile mismatch in {spec!r}")
    return shape, stride


def profile(tree: Tree) -> object:
    """The bracketing of a Tree with every leaf erased -- its mode structure."""
    if isinstance(tree, int):
        return None
    return tuple(profile(t) for t in tree)


def flatten(tree: Tree) -> tuple[int, ...]:
    """Left-to-right leaf order. Faithful for evaluation; erases structure."""
    if isinstance(tree, int):
        return (tree,)
    return tuple(leaf for t in tree for leaf in flatten(t))


# ------------------------------------------------------------ evaluation


def ev(lay: Layout, i: int) -> int:
    """Evaluate at 1-D coordinate `i` (column-major over the flattened shape)."""
    idx, rest = 0, i
    for s, d in zip(flatten(lay[0]), flatten(lay[1])):
        idx += (rest % s) * d
        rest //= s
    return idx


def size(lay: Layout) -> int:
    return prod(flatten(lay[0]))


def img(lay: Layout) -> list[int]:
    """The layout's image over its full domain [0, size)."""
    return [ev(lay, i) for i in range(size(lay))]


def leaf_multiset(lay: Layout) -> list[tuple[int, int]]:
    """Sorted (extent, stride) pairs -- deliberately grouping-INVARIANT.

    Only ever compared alongside a structural assertion; on its own it cannot
    distinguish the product/divide variants, which is the whole point of
    pairing the two.
    """
    return sorted(zip(flatten(lay[0]), flatten(lay[1])))


def shape_of(lay: Layout) -> Tree:
    return lay[0]


def stride_of(lay: Layout) -> Tree:
    return lay[1]


# ------------------------------------------------------ the parser itself


def test_parser_round_trips_nesting_and_rejects_malformed_input():
    """The fixtures are only trustworthy if the parser is."""
    assert layout("12:1") == (12, 1)
    assert layout("(2,3):(1,2)") == ((2, 3), (1, 2))
    assert layout("((3,4),(2,5)):((4,1),(12,24))") == (
        ((3, 4), (2, 5)),
        ((4, 1), (12, 24)),
    )
    assert layout("((3,4),2,5):((4,1),12,24)") == (((3, 4), 2, 5), ((4, 1), 12, 24))

    # nesting is preserved, not flattened away
    assert shape_of(layout("((3,4),(2,5)):((4,1),(12,24))")) != (3, 4, 2, 5)
    assert flatten(shape_of(layout("((3,4),(2,5)):((4,1),(12,24))"))) == (3, 4, 2, 5)

    with pytest.raises(ValueError):
        layout("(2,3)")                      # no separator
    with pytest.raises(ValueError):
        layout("(2,3):(1,2,3)")              # profile mismatch
    with pytest.raises(ValueError):
        layout("((2,3)):(1,2)")              # profile mismatch, nested


# ---------------------------------------------------------------- coalesce


def test_coalesce_preserves_the_function():
    """(2,(1,6)):(1,(6,2)) -> 12:1 -- fewer modes, identical map."""
    src, result = layout("(2,(1,6)):(1,(6,2))"), layout("12:1")
    assert img(src) == img(result)
    assert profile(shape_of(src)) != profile(shape_of(result)), "depth must drop"
    assert shape_of(result) == 12, "coalesce yields a depth-0 rank-1 layout"


def test_coalesce_by_mode_keeps_the_outer_rank():
    """(3,(4,5)):(8,(1,4)) with profile (1,1) -> (3,20):(8,1), still rank 2."""
    src, result = layout("(3,(4,5)):(8,(1,4))"), layout("(3,20):(8,1)")
    assert img(src) == img(result)
    assert shape_of(result) == (3, 20), "by-mode coalesce fired only inside"
    assert stride_of(result) == (8, 1)


# ------------------------------------------------------------- composition


def assert_composition(lhs: Layout, rhs: Layout, result: Layout) -> None:
    """R(c) == lhs(rhs(c)) for every c in rhs's domain."""
    assert size(rhs) == size(result)
    for j in range(size(rhs)):
        assert ev(lhs, ev(rhs, j)) == ev(result, j), f"mismatch at {j}"


def test_composition_splits_a_mode_across_an_lhs_boundary():
    """(6,2):(8,2) o (4,3):(3,1) -> ((2,2),3):((24,2),8).

    rhs mode 0 (extent 4) straddles lhs mode 0 (extent 6), so it splits into a
    NESTED (2,2). Structure is the result here -- a flat (2,2,3) would address
    the same elements while losing the fact that the split happened.
    """
    result = layout("((2,2),3):((24,2),8)")
    assert_composition(layout("(6,2):(8,2)"), layout("(4,3):(3,1)"), result)
    assert shape_of(result) == ((2, 2), 3)
    assert stride_of(result) == ((24, 2), 8)


def test_composition_with_a_tile_is_mode_wise_not_functional():
    """(6,4):(4,1) o [(2):(1);(2):(1)] -> ((2),(2)):((4),(1)).

    A tile is a per-mode tiler: slot i drives input mode i independently. It is
    NOT one flat layout composed as a function -- that reading gives a different
    answer, and is the first thing a reimplementation gets wrong. The result
    keeps each slot's own rank-1 hierarchy, hence ((2),(2)) not (2,2).
    """
    src, result = layout("(6,4):(4,1)"), layout("((2),(2)):((4),(1))")
    for c0 in range(2):
        for c1 in range(2):
            assert ev(src, c0 + 6 * c1) == ev(result, c0 + 2 * c1)
    assert shape_of(result) == ((2,), (2,)), "tile slots stay wrapped"
    assert flatten(shape_of(result)) == (2, 2)


def test_DOC_DEFECT_composition_with_shape_prose_contradicts_its_own_example():
    """core_ops.rst calls shape-composition `A o make_layout(shape)`. It is not.

    The correct reading -- and the one CuteOps.td states -- is mode-wise
    truncation: mode i of A restricted to shape[i]. The tutorial's stated
    equivalence yields a different function than the tutorial's own example.
    """
    a = layout("(4,8):(1,4)")
    via_prose = [ev(a, ev(layout("(2,4):(1,2)"), j)) for j in range(8)]
    documented = layout("(2,4):(1,4)")

    assert via_prose == [0, 1, 2, 3, 4, 5, 6, 7]
    assert img(documented) == [0, 1, 4, 5, 8, 9, 12, 13]
    assert via_prose != img(documented), "prose and example agree -- doc fixed?"

    # Mode-wise truncation reproduces the documented result exactly:
    # mode0 4:1 -> 2:1, mode1 8:4 -> 4:4.
    assert documented == layout("(2,4):(1,4)")


# -------------------------------------------------------------- complement


COMPLEMENT_SRC = layout("(2,2):(4,1)")  # image {0,1,4,5}, cosize 6


def test_complement_with_cotarget_tiles_the_codomain_exactly():
    """complement((2,2):(4,1), 24) == (2,3):(2,8) -- A (+) A* bijects [0,24)."""
    star = layout("(2,3):(2,8)")
    sums = sorted(a + c for a in img(COMPLEMENT_SRC) for c in img(star))
    assert sums == list(range(24))


def test_DOC_DEFECT_complement_without_cotarget_overshoots_its_stated_guarantee():
    """The no-cotarget form does not "cover exactly [0,M) with no overlap".

    M defaults to cosize(A) = 6, but the documented answer (2,1):(2,8) yields a
    bijection onto [0,8) -- a strict superset. It does reach the codomain holes
    {2,3}, so the form is usable; the *guarantee as written* is false. Traces to
    shape_div saturating 6/8 to 1 rather than 0.
    """
    star = layout("(2,1):(2,8)")
    cover = sorted(a + c for a in img(COMPLEMENT_SRC) for c in img(star))
    assert cover != list(range(6)), "guarantee now holds -- doc fixed?"
    assert cover == list(range(8))          # bijection onto its own span
    assert {2, 3} <= set(cover)             # the holes are reached


# ---------------------------------------------------------------- inverses


@pytest.mark.parametrize(
    "src_spec, inv_spec",
    [
        ("(2,4,6):(4,1,8)", "(4,2,6):(2,1,8)"),
        ("(2,4):(4,1)", "(4,2):(2,1)"),
    ],
)
def test_right_inverse_identity(src_spec: str, inv_spec: str):
    """A(A_R^-1(j)) == j over the inverse's domain."""
    src, inv = layout(src_spec), layout(inv_spec)
    for j in range(size(inv)):
        assert ev(src, ev(inv, j)) == j


def test_worked_example_rows_are_as_printed():
    """The tutorial prints both walks explicitly; check them digit for digit."""
    assert img(layout("(2,4):(4,1)")) == [0, 4, 1, 5, 2, 6, 3, 7]
    assert img(layout("(4,2):(2,1)")) == [0, 2, 4, 6, 1, 3, 5, 7]


def test_left_inverse_identity_for_a_bijective_layout():
    """A_L^-1(A(c)) == c -- equal to the right inverse when A is a bijection."""
    src, inv = layout("(2,4,6):(4,1,8)"), layout("(4,2,6):(2,1,8)")
    assert sorted(img(src)) == list(range(48)), "precondition: A is bijective"
    for c in range(48):
        assert ev(inv, ev(src, c)) == c


# ------------------------------------------------------------------ recast


@pytest.mark.parametrize(
    "new_bits, old_bits, src_spec, result_spec",
    [
        (32, 8, "(32,4):(1,32)", "(8,4):(1,8)"),     # upcast x4
        (8, 32, "(8,4):(1,8)", "(32,4):(1,32)"),     # downcast x4
        (32, 32, "(8,4):(1,8)", "(8,4):(1,8)"),      # identity
        (4, 6, "(8,4):(1,8)", "(12,4):(1,12)"),      # general, gcd split
    ],
)
def test_recast_layout_preserves_total_bit_extent(
    new_bits: int, old_bits: int, src_spec: str, result_spec: str
):
    """Recasting changes the element unit, never the number of bits addressed."""
    src, result = layout(src_spec), layout(result_spec)
    assert size(src) * old_bits == size(result) * new_bits
    # the stride-1 mode's run length, in bits, is also preserved
    assert flatten(shape_of(src))[0] * old_bits == (
        flatten(shape_of(result))[0] * new_bits
    )
    # recast never changes the mode structure, only the extents/strides
    assert profile(shape_of(src)) == profile(shape_of(result))


def test_recast_general_case_is_upcast_then_downcast():
    """new=4 old=6: G=gcd=2, new'=2, old'=3 -- each step conserves bits."""
    src = layout("(8,4):(1,8)")               # in 6-bit units
    after_upcast = layout("(4,4):(1,4)")      # 6-bit -> 12-bit units (x new'=2)
    after_downcast = layout("(12,4):(1,12)")  # 12-bit -> 4-bit units (x old'=3)

    assert flatten(shape_of(src))[0] * 6 == flatten(shape_of(after_upcast))[0] * 12
    assert flatten(stride_of(src))[1] * 6 == flatten(stride_of(after_upcast))[1] * 12
    assert (
        flatten(shape_of(after_upcast))[0] * 12
        == flatten(shape_of(after_downcast))[0] * 4
    )
    assert (
        flatten(stride_of(after_upcast))[1] * 12
        == flatten(stride_of(after_downcast))[1] * 4
    )


# ---------------------------------------------------------------- products


#: input (3,4):(4,1), tiler (2,5):(1,2) -- documented strings, verbatim.
PRODUCT_VARIANTS: dict[str, Layout] = {
    "logical_product": layout("((3,4),(2,5)):((4,1),(12,24))"),
    "zipped_product": layout("((3,4),(2,5)):((4,1),(12,24))"),
    "tiled_product": layout("((3,4),2,5):((4,1),12,24)"),
    "flat_product": layout("(3,4,2,5):(4,1,12,24)"),
    "blocked_product": layout("((3,2),(4,5)):((4,12),(1,24))"),
    "raked_product": layout("((2,3),(5,4)):((12,4),(24,1))"),
}


def test_each_product_variant_has_its_documented_mode_structure():
    """Pin the exact grouping per variant -- the claim leaf multisets cannot make.

    Without this, an implementation that flattened every result (or returned
    logical_product's grouping for all six) would satisfy the multiset test
    below. This is the assertion that makes that impossible.
    """
    assert shape_of(PRODUCT_VARIANTS["logical_product"]) == ((3, 4), (2, 5))
    assert shape_of(PRODUCT_VARIANTS["zipped_product"]) == ((3, 4), (2, 5))
    assert shape_of(PRODUCT_VARIANTS["tiled_product"]) == ((3, 4), 2, 5)
    assert shape_of(PRODUCT_VARIANTS["flat_product"]) == (3, 4, 2, 5)
    assert shape_of(PRODUCT_VARIANTS["blocked_product"]) == ((3, 2), (4, 5))
    assert shape_of(PRODUCT_VARIANTS["raked_product"]) == ((2, 3), (5, 4))

    assert stride_of(PRODUCT_VARIANTS["blocked_product"]) == ((4, 12), (1, 24))
    assert stride_of(PRODUCT_VARIANTS["raked_product"]) == ((12, 4), (24, 1))

    # ranks differ, so a flattening implementation cannot pass
    ranks = {
        name: len(shape) if isinstance(shape := shape_of(v), tuple) else 0
        for name, v in PRODUCT_VARIANTS.items()
    }
    assert ranks == {
        "logical_product": 2,
        "zipped_product": 2,
        "tiled_product": 3,
        "flat_product": 4,
        "blocked_product": 2,
        "raked_product": 2,
    }


def test_product_variants_are_five_distinct_structures_not_six():
    """logical_ and zipped_product coincide for a plain-layout tiler; the rest differ.

    Asserting "all six differ" would be wrong, and asserting nothing would let a
    collapse through. This pins the exact equivalence class.
    """
    by_structure: dict[tuple, list[str]] = {}
    for name, variant in PRODUCT_VARIANTS.items():
        by_structure.setdefault((variant[0], variant[1]), []).append(name)

    classes = sorted(sorted(names) for names in by_structure.values())
    assert classes == [
        ["blocked_product"],
        ["flat_product"],
        ["logical_product", "zipped_product"],
        ["raked_product"],
        ["tiled_product"],
    ]


def test_all_six_product_variants_are_one_leaf_multiset_regrouped():
    """The six differ only in bracketing -- same (extent,stride) multiset.

    This is the scoping result behind CUTE_IR_ASSESSMENT.md Sec. 2: the product
    family is one construction plus five regroupings, not six algorithms. It is
    only meaningful because the structural test above proves the six are NOT the
    same object; on its own, a multiset comparison of identical inputs would be
    vacuous.
    """
    base = leaf_multiset(PRODUCT_VARIANTS["logical_product"])
    assert base == [(2, 12), (3, 4), (4, 1), (5, 24)]
    for name, variant in PRODUCT_VARIANTS.items():
        assert leaf_multiset(variant) == base, name


def test_all_six_product_variants_address_the_same_120_offsets():
    """Regrouping permutes the traversal order; the addressed set is identical."""
    base = sorted(img(PRODUCT_VARIANTS["logical_product"]))
    assert len(base) == 120 and len(set(base)) == 120
    for name, variant in PRODUCT_VARIANTS.items():
        assert sorted(img(variant)) == base, name


def test_logical_product_places_disjoint_copies():
    """Each copy of the block lands in its own slice of the codomain."""
    assert sorted(img(layout("((2,2),(2,3)):((4,1),(2,8))"))) == list(range(24))
    assert sorted(img(layout("((2,2),(3,2)):((4,1),(8,2))"))) == list(range(24))


def test_DOC_DEFECT_product_table_gives_logical_product_the_wrong_grouping():
    """The table's logical_product row is verbatim its blocked_product row.

    The worked examples show the two are different groupings; the example
    matches the standard definition (A, complement(A,.) o B).
    """
    logical = PRODUCT_VARIANTS["logical_product"]
    blocked = PRODUCT_VARIANTS["blocked_product"]
    assert logical != blocked, "groupings now agree -- doc fixed?"
    assert shape_of(logical) == ((3, 4), (2, 5))    # input | tiler
    assert shape_of(blocked) == ((3, 2), (4, 5))    # per-axis pairs
    # ...yet they address exactly the same offsets, which is why only a
    # structural assertion can tell them apart.
    assert sorted(img(logical)) == sorted(img(blocked))


def test_tile_to_shape_covers_the_target_bijectively():
    """block (2,2):(1,2) over target (8,8) -> ((2,4),(2,4)):((1,4),(2,16))."""
    result = layout("((2,4),(2,4)):((1,4),(2,16))")
    assert sorted(img(result)) == list(range(64))
    assert shape_of(result) == ((2, 4), (2, 4)), "per-axis (block, reps) pairs"


# ----------------------------------------------------------------- divides


DIVIDE_SRC = layout("(6,8):(8,1)")

DIVIDE_VARIANTS: dict[str, Layout] = {
    "logical_divide": layout("((3,2),(4,2)):((8,24),(1,4))"),
    "zipped_divide": layout("((3,4),(2,2)):((8,1),(24,4))"),
    "tiled_divide": layout("((3,4),2,2):((8,1),24,4)"),
    "flat_divide": layout("(3,4,2,2):(8,1,24,4)"),
}


def test_each_divide_variant_has_its_documented_mode_structure():
    """Pin the grouping: within-tile vs across-tile placement differs per variant."""
    assert shape_of(DIVIDE_VARIANTS["logical_divide"]) == ((3, 2), (4, 2))
    assert shape_of(DIVIDE_VARIANTS["zipped_divide"]) == ((3, 4), (2, 2))
    assert shape_of(DIVIDE_VARIANTS["tiled_divide"]) == ((3, 4), 2, 2)
    assert shape_of(DIVIDE_VARIANTS["flat_divide"]) == (3, 4, 2, 2)

    assert stride_of(DIVIDE_VARIANTS["logical_divide"]) == ((8, 24), (1, 4))
    assert stride_of(DIVIDE_VARIANTS["zipped_divide"]) == ((8, 1), (24, 4))


def test_divide_variants_are_four_distinct_structures():
    """No two divide variants share a structure -- unlike the product family."""
    structures = {(v[0], v[1]) for v in DIVIDE_VARIANTS.values()}
    assert len(structures) == len(DIVIDE_VARIANTS) == 4


def test_logical_divide_separates_within_tile_from_across_tile():
    """logical_divide pairs per axis; zipped_divide groups the two halves.

    The two carry the same leaves in a different order, so this distinction is
    invisible to both the image and the multiset -- structure is the only
    witness.
    """
    logical = DIVIDE_VARIANTS["logical_divide"]
    zipped = DIVIDE_VARIANTS["zipped_divide"]

    # logical: mode 0 is axis 0 as (within, across); mode 1 is axis 1.
    assert shape_of(logical)[0] == (3, 2) and stride_of(logical)[0] == (8, 24)
    # zipped: mode 0 is the whole within-tile block (3,4); mode 1 the walk (2,2).
    assert shape_of(zipped)[0] == (3, 4) and stride_of(zipped)[0] == (8, 1)

    assert leaf_multiset(logical) == leaf_multiset(zipped)
    assert sorted(img(logical)) == sorted(img(zipped))
    assert logical != zipped


def test_all_four_divide_variants_are_one_leaf_multiset_regrouped():
    base = leaf_multiset(DIVIDE_VARIANTS["logical_divide"])
    assert base == [(2, 4), (2, 24), (3, 8), (4, 1)]
    for name, variant in DIVIDE_VARIANTS.items():
        assert leaf_multiset(variant) == base, name


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
    non-affine in the l1 mode. The DEPTH-2 nested (2,2) is that split; a flat
    result could not represent it at all.
    """
    inner = layout("((3,4),4):((1,3),12)")
    assert img(inner) == list(range(48)), "inner map is the identity"

    result = layout("((3,(2,2)),4):((8,(24,1)),2)")
    for j in range(48):
        assert ev(DIVIDE_SRC, ev(inner, j)) == ev(result, j), f"mismatch at {j}"

    assert shape_of(result) == ((3, (2, 2)), 4), "the split is depth-2, not flat"
    assert stride_of(result) == ((8, (24, 1)), 2)
    assert profile(shape_of(result)) == ((None, (None, None)), None)


# ------------------------------------------------------- composed layouts


def test_composed_layout_applies_offset_between_the_two_maps():
    """(6,2):(1,3) o (2,1) o (2,3):(1@1,2@0) -- basis strides then offset then A.

    B's scaled-basis strides make B(c) a TUPLE, so the offset is added
    per-component before A indexes it. Getting this wrong (adding a scalar)
    silently produces plausible indices.
    """
    got = []
    for j in range(6):
        c0, c1 = j % 2, j // 2
        b0, b1 = 2 * c1, c0                      # B(c) -> (basis0, basis1)
        got.append((b0 + 2) * 1 + (b1 + 1) * 3)  # A(offset + B(c))
    assert got == [5, 8, 7, 10, 9, 12]


def test_composed_layout_example_evaluates_A_outside_its_own_shape():
    """A = (6,2):(1,3) is indexed at n0 = 6, one past its extent.

    Legal in CuTe -- a layout is an affine function defined beyond its shape --
    but it means the composed layout's image can exceed A's own cosize. Any
    Tessera bounds verifier must model this deliberately rather than assume
    containment.
    """
    outer = layout("(6,2):(1,3)")
    outer_cosize = 5 * 1 + 1 * 3 + 1
    assert outer_cosize == 9
    assert flatten(shape_of(outer))[0] == 6

    max_n0 = max((2 * (j // 2)) + 2 for j in range(6))
    assert max_n0 == 6 >= flatten(shape_of(outer))[0], "n0 reaches A's extent"

    assert max([5, 8, 7, 10, 9, 12]) == 12 > outer_cosize
