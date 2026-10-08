"""Bounded projections keep every traced result-type cache coherent."""
import pytest
from tessera.compiler import scheduled_matmul as scheduled
from tests.unit.test_scheduled_matmul_consumers import _module


@pytest.mark.parametrize("storage", ["fp16", "bf16"])
@pytest.mark.parametrize("axes", [("M",), ("M", "K"), ("M", "N", "K")])
def test_bounded_projection_updates_traced_result_tuple_without_mutating_input(storage, axes):
    graph = _module(target="nvidia_sm120", shape=(16, 16, 8), dtype=storage)
    op = graph.functions[0].body[0]
    # Normal tracing fills the plural cache even for a one-result operation.
    original = graph.functions[0].result_types[0]
    op.inferred_type = original
    op.inferred_types = (original,)
    before = graph.to_mlir(target="nvidia_sm120", canonical=True)
    if axes == ("M",):
        projected = scheduled.with_bounded_dynamic_m(graph, 16)
    elif axes == ("M", "K"):
        projected = scheduled.with_bounded_dynamic_mk(graph, 16, 16)
    else:
        projected = scheduled.with_bounded_dynamic_axes(graph, axes)
    function = projected.functions[0]
    result = function.result_types[0]
    operation = function.body[0]
    assert operation.inferred_type == result
    assert operation.inferred_types == (result,)
    assert operation.result_type == str(result)
    text = projected.to_mlir(target="nvidia_sm120", canonical=True)
    declaration = next(line for line in text.splitlines() if "tessera.matmul" in line)
    assert "-> " + str(result) in declaration
    assert graph.to_mlir(target="nvidia_sm120", canonical=True) == before
