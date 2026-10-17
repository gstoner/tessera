"""Captured coordinates are literals, never executable/tensor attributes."""
import ast
import pytest
import tessera as ts
from tessera.compiler.graph_ir import GraphIRBuilder, _OpExtractor


def factory(rows, width, keyword):
    if keyword:
        def slices(parent):
            left = ts.ops.slice(parent, start_indices=[0, 0], slice_sizes=(rows, width))
            right = ts.ops.slice(parent, start_indices=[0, width], slice_sizes=(rows, width))
            return ts.ops.silu_mul(left, right)
    else:
        def slices(parent):
            left = ts.ops.slice(parent, [0, 0], [rows, width])
            right = ts.ops.slice(parent, [0, width], [rows, width])
            return ts.ops.silu_mul(left, right)
    return slices


@pytest.mark.parametrize("keyword", [False, True])
def test_nested_closure_coordinates_reach_original_graph_attributes(keyword):
    first = GraphIRBuilder().lower(factory(3, 17, keyword))
    second = GraphIRBuilder().lower(factory(16, 64, keyword))
    assert [op.op_name for op in first.body] == ["tessera.slice", "tessera.slice", "tessera.silu_mul"]
    assert first.body[0].kwargs["slice_sizes"] == [3, 17]
    assert first.body[1].kwargs["start_indices"] == [0, 17]
    assert second.body[0].kwargs["slice_sizes"] == [16, 64]
    assert second.body[1].kwargs["start_indices"] == [0, 64]
    assert len(first.body[0].operands) == len(first.body[1].operands) == 1


def test_nested_attribute_capture_copies_values_and_preserves_tensor_shadowing():
    sizes = [3, 17]
    extractor = _OpExtractor(["parent", "width"], const_env={"sizes": sizes, "width": 64})
    literal = extractor._attribute_literal(ast.parse("{'shape': sizes}", mode="eval").body)
    literal["shape"][0] = 99
    assert sizes == [3, 17]
    with pytest.raises(ValueError, match="literal"):
        extractor._attribute_literal(ast.parse("[0, width]", mode="eval").body)


@pytest.mark.parametrize("expression", ["[value()]", "[sizes[0]]", "[obj.extent]", "[width + 1]"])
def test_captured_coordinate_expressions_are_never_executed(expression):
    extractor = _OpExtractor([], const_env={"width": 17, "sizes": [3, 17]})
    with pytest.raises(ValueError, match="literal"):
        extractor._attribute_literal(ast.parse(expression, mode="eval").body)
