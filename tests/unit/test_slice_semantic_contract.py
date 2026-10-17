"""Public slicing semantics must survive abstract tracing and native Graph verification."""

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

CASES = [
    ((4, 4), (3, 3), (4, 4)),
    ((4, 4), (-1, 0), (1, 3)),
    ((4, 4), (2, 0), (-3, 4)),
    ((3, 34), (0, 17), (3, 17)),
    ((4, 4), (0, 9), (4, 2)),
    ((4, 4), (0, 0), (0, 4)),
]


@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
@pytest.mark.parametrize("shape,starts,sizes", CASES)
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("oracle", [False, True])
def test_public_slice_trace_shape_matches_eager(name, shape, starts, sizes, positional, oracle):
    def fn(x):
        op = getattr(ts.ops, name)
        if positional:
            return op(x, list(starts), list(sizes))
        return op(x, start_indices=list(starts), slice_sizes=list(sizes))

    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = fn(source)
    for spec in [source, (shape, "f32")]:
        traced = trace(fn, spec, evaluate_catalog_outputs=oracle)
        assert traced.output_specs[0] == (expected.shape, "f32")
        graph = to_graph_ir_module(traced, name="slice_trace", target="rocm_gfx1201")
        body = graph.functions[0].body
        assert len(body) == 1 and body[0].op_name == "tessera." + name
        assert body[0].kwargs["start_indices"] == list(starts)
        assert body[0].kwargs["slice_sizes"] == list(sizes)


@pytest.fixture(scope="module")
def tool():
    value = find_tessera_opt()
    assert value is not None, "Requires production compiler"
    return value


def module(name, *, result="1x1xf32", starts="3, 3", sizes="4, 4"):
    return f"""module {{func.func @slice(%x: tensor<4x4xf32>) -> tensor<{result}> {{
      %y = "tessera.{name}"(%x) {{start_indices=[{starts}], slice_sizes=[{sizes}]}} : (tensor<4x4xf32>) -> tensor<{result}>
      return %y : tensor<{result}>
    }} }}"""


@pytest.mark.compiler_route
@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
def test_registered_slice_verifies_clipped_output(tool, name):
    text = run_tessera_opt(tool, module(name), "--verify-each")
    assert "tensor<1x1xf32>" in text


@pytest.mark.compiler_route
@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
@pytest.mark.parametrize(
    "options,diagnostic",
    [
        ({"result": "4x4xf32"}, "normalized slice extent"),
        ({"starts": "3"}, "match input rank"),
        ({"sizes": "4"}, "match input rank"),
        ({"result": "1x1xf16"}, "same element type"),
    ],
)
def test_registered_slice_rejects_invalid_contract(tool, name, options, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        run_tessera_opt(tool, module(name, **options), "--verify-each")


@pytest.mark.compiler_route
@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
@pytest.mark.parametrize(
    "mode,flag,expected",
    [
        ("forward", "--tessera-autodiff-forward", "tessera."),
        ("reverse", "--tessera-autodiff-paired", "tensor.insert_slice"),
    ],
)
def test_slice_native_linear_ad(tool, name, mode, flag, expected):
    source = module(name).replace(
        "> {\n", '> attributes {tessera.autodiff = "' + mode + '", tessera.autodiff.wrt_indices = [0]} {\n', 1
    )
    text = run_tessera_opt(tool, source, flag)
    run_tessera_opt(tool, text, "--verify-each")
    assert expected in text
    if mode == "forward":
        assert text.count("tessera." + name) >= 2
    else:
        assert "dense<0.000000e+00>" in text


@pytest.mark.compiler_route
@pytest.mark.parametrize(
    "extent,start,size",
    [
        (4, (1 << 63) - 1, (1 << 63) - 1),
        (4, -(1 << 63), -(1 << 63)),
        ((1 << 63) - 1, (1 << 63) - 2, 8),
        ((1 << 63) - 1, -(1 << 63), (1 << 63) - 1),
    ],
)
def test_slice_native_verifier_widens_end_arithmetic(tool, extent, start, size):
    first, end, _ = slice(start, start + size).indices(extent)
    count = max(0, end - first)
    source = f"""module {{func.func @wide(%x:tensor<{extent}xf32>) -> tensor<{count}xf32> {{
      %y = "tessera.slice"(%x) {{start_indices=[{start}],slice_sizes=[{size}]}} : (tensor<{extent}xf32>) -> tensor<{count}xf32>
      return %y : tensor<{count}xf32>
    }} }}"""
    text = run_tessera_opt(tool, source, "--verify-each")
    assert "tessera.slice" in text


@pytest.mark.parametrize("start", [-(1 << 63) - 1, 1 << 63])
def test_trace_rejects_coordinate_wrap_before_mlir(start):
    def fn(x):
        return ts.ops.slice(x, start_indices=[start], slice_sizes=[2])

    for oracle in [False, True]:
        with pytest.raises(ValueError, match="signed i64"):
            trace(fn, np.ones((4,), dtype=np.float32), evaluate_catalog_outputs=oracle)


def test_trace_preserves_eager_coordinate_integer_coercion():
    def fn(x):
        return ts.ops.slice(x, start_indices=[True, 1.75], slice_sizes=[2.9, 3.25])

    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    expected = fn(source)
    for spec in [source, ((4, 4), "f32")]:
        traced = trace(fn, spec)
        assert traced.output_specs[0] == (expected.shape, "f32")
        assert traced.body[0].kwargs == {"start_indices": [1, 1], "slice_sizes": [2, 3]}


@pytest.mark.compiler_route
@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
def test_slice_transpose_guards_dynamic_cotangent_extents(tool, name):
    source = module(name, result="?x?xf32").replace(
        "> {\n", '> attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [0]} {\n', 1
    )
    text = run_tessera_opt(tool, source, "--tessera-autodiff-paired")
    run_tessera_opt(tool, text, "--verify-each")
    assert text.count("slice cotangent shape mismatch") == 2
    assert "cf.assert" in text and "tensor.insert_slice" in text


@pytest.mark.compiler_route
@pytest.mark.parametrize("name", ["slice", "dynamic_slice"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dynamic_result", [False, True])
def test_slice_and_reverse_reach_standard_structured_tensor_ir(tool, name, reverse, dynamic_result):
    source = module(name, result="?x?xf32" if dynamic_result else "1x1xf32")
    if reverse:
        source = source.replace(
            "> {\n", '> attributes {tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [0]} {\n', 1
        )
        source = run_tessera_opt(tool, source, "--tessera-autodiff-paired")
    text = run_tessera_opt(tool, source, "--tessera-to-linalg")
    run_tessera_opt(tool, text, "--verify-each")
    assert "tessera.slice" not in text and "tessera.dynamic_slice" not in text
    assert "tensor.extract_slice" in text
    assert "[3, 3] [1, 1] [1, 1]" in text
    if reverse:
        assert "tensor.insert_slice" in text


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_concrete_slice_preserves_non_native_silu_promotion(dtype):
    def fn(x):
        gate = ts.ops.slice(x, [0, 0], [2, 2])
        up = ts.ops.slice(x, [0, 2], [2, 2])
        return ts.ops.silu_mul(gate, up)

    source = np.arange(8, dtype=dtype).reshape(2, 4)
    expected = np.asarray(fn(source))
    traced = trace(fn, source)
    assert traced.output_values[0] is not None
    np.testing.assert_array_equal(traced.output_values[0], expected)
    assert np.dtype(traced.output_values[0].dtype) == expected.dtype


def test_concrete_slice_carries_a_borrowed_view_without_compaction():
    def fn(x):
        return ts.ops.slice(x, [0, 1], [3, 2])

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    traced = trace(fn, source)
    view = traced.output_values[0]
    assert view is not None and np.shares_memory(view, source)
    assert view.strides == source.strides and not view.flags.c_contiguous
    np.testing.assert_array_equal(view, source[:, 1:3])
