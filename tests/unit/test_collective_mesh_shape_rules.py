"""A collective's result shape is a function of the MESH, and must fail closed.

`all_gather` and `reduce_scatter` were recorded as deliberately undeclared:
"result shape scales with mesh size, not derivable from operand types". The
premise is right; the conclusion does not follow. Not derivable from *operand
types* is an argument for giving the rule signature mesh context — which is
what `attrs` already did one level down for `reshape`/`cast` — not for leaving
the ops unclassified.

And leaving them unclassified was not neutral. `unclassified` resolves to
same-as-first, so both ops returned the operand's own shape: the positive claim
that ``world_size == 1``. The reference collectives are single-rank no-op stubs,
so a probe agreed and the wrong rule looked confirmed. That is the same
degenerate-probe trap that produced the wrong `ebm_self_verify` rule, arriving
by a different route — measuring a system at the one configuration where the
wrong answer and the right answer coincide.
"""

from __future__ import annotations

import pytest

from tessera.compiler.graph_ir import _infer_result_type, tensor_ir_type
from tessera.compiler.op_catalog import DELIBERATELY_UNDECLARED, shape_rule_for

_X = tensor_ir_type(("8", "16"), "bf16")


def _infer(op, attrs, mesh=None):
    return _infer_result_type(f"tessera.{op}", [_X], attrs, mesh=mesh)


@pytest.mark.parametrize("op", ["all_gather", "reduce_scatter"])
def test_unknown_mesh_yields_an_unknown_axis_not_the_operand_extent(op):
    """The core invariant. `?` is the honest answer; `8` is a claim.

    This is the assertion that would have failed on the old behaviour, and the
    reason it is phrased as "must NOT equal the operand" rather than "must
    equal ?": any answer that silently reuses the operand extent is wrong,
    whatever it is spelled.
    """
    result = _infer(op, {"axis": "dp"})
    assert str(result) != str(_X), (
        f"{op} with no mesh returned the operand shape {result} — that is the "
        "claim world_size == 1, not an absence of information"
    )
    assert result.shape[0] == "?", result
    assert result.shape[1] == "16", "unscaled axes must be preserved"


def test_all_gather_multiplies_the_scaled_axis_by_the_mesh_extent():
    assert str(_infer("all_gather", {"axis": "dp"}, {"dp": 4})) == "tensor<32x16xbf16>"


def test_reduce_scatter_divides_the_scaled_axis_by_the_mesh_extent():
    assert str(_infer("reduce_scatter", {"axis": "dp"}, {"dp": 4})) == "tensor<2x16xbf16>"


def test_reduce_scatter_fails_closed_on_a_non_divisible_mesh():
    """8 tokens over 3 ranks has no uniform answer, so it must not invent one.

    Integer division would have produced `2` — plausible, checkable, and off by
    two elements per gather.
    """
    result = _infer("reduce_scatter", {"axis": "dp"}, {"dp": 3})
    assert result.shape[0] == "?", result


def test_integer_axis_names_a_tensor_axis_not_a_mesh_axis():
    """`axis: int | str` is overloaded: int = tensor axis, str = mesh axis.

    With an int the mesh axis is unnamed, so the extent is genuinely unknown
    even when a mesh is supplied — and the rule says `?` rather than reaching
    for whichever mesh axis happens to be present.
    """
    result = _infer("all_gather", {"axis": 1}, {"dp": 4})
    assert result.shape == ("8", "?"), result


@pytest.mark.parametrize("op", ["all_reduce", "all_to_all"])
def test_shape_preserving_collectives_are_not_mesh_scaled(op):
    """Not every collective scales. `all_reduce` and `all_to_all` preserve
    shape, so their `same_as_first` default is already correct — declaring them
    mesh-aware would be a wrong rule dressed as thoroughness."""
    assert str(_infer(op, {"axis": "dp"}, {"dp": 4})) == str(_X)


@pytest.mark.parametrize("op", ["all_gather", "reduce_scatter"])
def test_exemptions_are_retired(op):
    graph_name = f"tessera.{op}"
    assert graph_name not in DELIBERATELY_UNDECLARED
    assert shape_rule_for(graph_name) == op
