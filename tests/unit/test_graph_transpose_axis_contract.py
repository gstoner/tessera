"""Native Graph transpose semantics for mapped output-axis integration."""
from __future__ import annotations
import os
import shutil
import subprocess

import pytest
import tessera as ts


@pytest.fixture
def opt():
    tool = os.environ.get("TESSERA_OPT") or shutil.which("tessera-opt")
    if tool is None:
        pytest.skip("matching production tessera-opt required")
    return tool


def module(source, result, attributes=""):
    attrs = (" {" + attributes + "}") if attributes else ""
    return (
        "module { func.func @axis(%x: tensor<" + source + ">) -> tensor<" + result + "> {\n"
        ' %y = "tessera.transpose"(%x)' + attrs +
        " : (tensor<" + source + ">) -> tensor<" + result + ">\n"
        " return %y : tensor<" + result + ">\n} }\n"
    )


def invoke(opt, source, *flags):
    return subprocess.run([opt, *flags], input=source, text=True,
                          capture_output=True, timeout=60)


@pytest.mark.parametrize(("source", "result", "attributes"), [
    ("2x3xf32", "3x2xf32", ""),
    ("2x3x5xf32", "5x3x2xf32", ""),
    ("2x3x5xf32", "3x2x5xf32", "permutation = array<i64: 1, 0, 2>"),
    ("2x3x5xf32", "2x3x5xf32", "permutation = array<i64: 0, 1, 2>"),
    ("3x3xf32", "3x3xf32", "permutation = array<i64: 1, 0>"),
    ("?x3x5xf32", "3x7x5xf32", "permutation = array<i64: 1, 0, 2>"),
    ("2x3x5xf32", "?x2x5xf32", "permutation = array<i64: 1, 0, 2>"),
    ("2x3x5xf32", "3x2x5xf32",
     'permutation = array<i64: 1, 0, 2>, tessera.result_view_policy = "preserve"'),
])
def test_graph_transpose_accepts_axis_exact_results(opt, source, result, attributes):
    compiled = invoke(opt, module(source, result, attributes))
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize(("result", "attributes"), [
    ("2x3x5xf32", "permutation = array<i64: 1, 0, 2>"),
    ("3x5x2xf32", "permutation = array<i64: 1, 0, 2>"),
    ("2x3x5xf32", ""),
    ("2x3x5xf32", "permutation = array<i64: 0, 0, 2>"),
    ("2x3x5xf32", "permutation = array<i64: 0, 1, 3>"),
    ("2x3x5xf32", "permutation = array<i64: 0, 1, -1>"),
    ("2x3x5xf32", "permutation = array<i64: 0, 1>"),
    ("2x3x5xf32", "permutation = array<i64: 0, 1, 2, 3>"),
    ("2x3x5xf32", "permutation = [0 : i64, 1 : i64, 2 : i64]"),
    ("2x3x5xf32", 'permutation = "identity"'),
    ("5x3x2xf32", "axes = [2 : i64, 0 : i64, 1 : i64]"),
    ("5x3x2xf32", "perm = array<i64: 2, 0, 1>"),
    ("5x3x2xf32", "tessera.perm = [2 : i64, 0 : i64, 1 : i64]"),
])
def test_graph_transpose_rejects_malformed_axes_before_lowering(opt, result, attributes):
    compiled = invoke(opt, module("2x3x5xf32", result, attributes))
    assert compiled.returncode != 0, compiled.stdout
    assert "requires a rank-sized unique nonnegative i64 permutation" in compiled.stderr


def test_identity_metadata_is_preserved_even_with_valid_semantic_axes(opt):
    source = module("2x3x5xf32", "2x3x5xf32",
                    'permutation = array<i64: 0, 1, 2>, tessera.result_view_policy = "preserve"')
    compiled = invoke(opt, source, "--canonicalize")
    assert compiled.returncode == 0, compiled.stderr
    assert "tessera.transpose" in compiled.stdout
    assert 'tessera.result_view_policy = "preserve"' in compiled.stdout


def test_plain_identity_can_fold_after_axis_verification(opt):
    compiled = invoke(opt, module("2x3x5xf32", "2x3x5xf32",
                                  "permutation = array<i64: 0, 1, 2>"), "--canonicalize")
    assert compiled.returncode == 0, compiled.stderr
    assert "tessera.transpose" not in compiled.stdout


@pytest.mark.parametrize("shape", [(2, 3, 5), (3, 3, 3), (2, 3, 4, 5)])
def test_general_rank_transpose_materializes_in_native_linalg(opt, shape):
    permutation = tuple(range(1, len(shape))) + (0,)
    source_type = "x".join(map(str, shape)) + "xf32"
    result_type = "x".join(str(shape[axis]) for axis in permutation) + "xf32"
    attributes = "permutation = array<i64: " + ", ".join(map(str, permutation)) + ">"
    compiled = invoke(opt, module(source_type, result_type, attributes), "--tessera-to-linalg")
    assert compiled.returncode == 0, compiled.stderr
    assert "tessera.transpose" not in compiled.stdout
    assert "linalg.transpose" in compiled.stdout


def differentiated_source(mode):
    text = module("2x3x5xf32", "5x2x3xf32",
                  "permutation = array<i64: 2, 0, 1>")
    return text.replace(" {\n", ' attributes {tessera.autodiff = "' + mode +
                        '", tessera.autodiff.wrt_indices = [0]} {\n', 1)


def test_reverse_ad_uses_inverse_declared_axes(opt):
    compiled = invoke(opt, differentiated_source("reverse"), "--tessera-autodiff-paired")
    assert compiled.returncode == 0, compiled.stderr
    assert "permutation = array<i64: 1, 2, 0>" in compiled.stdout
    lowered = invoke(opt, compiled.stdout, "--tessera-to-linalg")
    assert lowered.returncode == 0, lowered.stderr
    assert "tessera.transpose" not in lowered.stdout


def test_forward_ad_retains_declared_axes(opt):
    compiled = invoke(opt, differentiated_source("forward"), "--tessera-autodiff-forward")
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout.count("permutation = array<i64: 2, 0, 1>") >= 2
    lowered = invoke(opt, compiled.stdout, "--tessera-to-linalg")
    assert lowered.returncode == 0, lowered.stderr
    assert "tessera.transpose" not in lowered.stdout


@pytest.mark.parametrize("shape", [(2, 3, 5), (3, 3, 3), (2, 3, 4, 5)])
def test_general_rank_permutation_executes_in_native_cpu_jit(opt, shape):
    from pathlib import Path
    import numpy as np
    if not Path(os.environ.get("TESSERA_JIT_LIB", "/missing")).is_file():
        pytest.skip("native CPU JIT required; artifact proof is insufficient")
    from tessera import _jit_boundary as jit
    permutation = tuple(range(1, len(shape))) + (0,)
    source_type = "x".join(map(str, shape)) + "xf32"
    result_shape = tuple(shape[axis] for axis in permutation)
    result_type = "x".join(map(str, result_shape)) + "xf32"
    attributes = "permutation = array<i64: " + ", ".join(map(str, permutation)) + ">"
    lowered = invoke(opt, module(source_type, result_type, attributes), "--tessera-to-linalg")
    assert lowered.returncode == 0, lowered.stderr
    value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    output = np.empty(result_shape, np.float32)
    before = jit.invocation_count()
    handle = jit.compile_module(lowered.stdout)
    try:
        jit.invoke(handle, "axis", [value], [output])
    finally:
        jit.destroy(handle)
    assert jit.invocation_count() == before + 1
    np.testing.assert_array_equal(output, value.transpose(permutation))


def test_inverse_permutation_ad_executes_in_native_cpu_jit(opt):
    from pathlib import Path
    import numpy as np
    if not Path(os.environ.get("TESSERA_JIT_LIB", "/missing")).is_file():
        pytest.skip("native CPU JIT required; artifact proof is insufficient")
    from tessera import _jit_boundary as jit
    differentiated = invoke(opt, differentiated_source("reverse"), "--tessera-autodiff-paired")
    assert differentiated.returncode == 0, differentiated.stderr
    lowered = invoke(opt, differentiated.stdout, "--tessera-to-linalg")
    assert lowered.returncode == 0, lowered.stderr
    assert "tessera.transpose" not in lowered.stdout
    value = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    seed = (np.arange(30, dtype=np.float32) * .125 - 1).reshape(5, 2, 3)
    output = np.empty_like(value)
    handle = jit.compile_module(lowered.stdout)
    before = jit.invocation_count()
    try:
        jit.invoke(handle, "axis__bwd", [value, seed], [output])
    finally:
        jit.destroy(handle)
    assert jit.invocation_count() == before + 1
    np.testing.assert_array_equal(output, seed.transpose(1, 2, 0))


def explicit_axis_source(x: ts.f32[2, 3, 5]):
    return ts.ops.transpose(x, axes=(2, 0, 1))


def positional_axis_source(x: ts.f32[2, 3, 5]):
    return ts.ops.transpose(x, (-1, 0, 1))


def reversed_axis_source(x: ts.f32[2, 3, 5]):
    return x.T


@pytest.mark.parametrize("function", [explicit_axis_source, positional_axis_source])
@pytest.mark.parametrize("frontend", ["ast", "tracer"])
def test_ordinary_source_axes_reach_verified_native_linalg(opt, function, frontend):
    from tessera.compiler.graph_ir import GraphIRBuilder
    from tessera.compiler.trace import trace, to_graph_ir_module
    if frontend == "ast":
        builder = GraphIRBuilder()
        builder.lower(function)
        graph = builder.context.module
    else:
        graph = to_graph_ir_module(trace(function, ((2, 3, 5), "fp32")),
                                   name=function.__name__)
    assert graph.functions[0].result_types[0].shape == ("5", "2", "3")
    source = graph.to_mlir(canonical=True)
    assert "permutation = array<i64: 2, 0, 1>" in source
    compiled = invoke(opt, source, "--tessera-to-linalg")
    assert compiled.returncode == 0, compiled.stderr
    assert "tessera.transpose" not in compiled.stdout
    assert "linalg.transpose" in compiled.stdout
    # When the native engine is present, execute this exact ordinary-source
    # Graph rather than rebuilding a manually typed numerical fixture.
    from pathlib import Path
    if Path(os.environ.get("TESSERA_JIT_LIB", "/missing")).is_file():
        import numpy as np
        from tessera import _jit_boundary as jit
        value = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
        handle = jit.compile_module(compiled.stdout)
        before = jit.invocation_count()
        compile_count = jit.compile_count()
        outputs = []
        try:
            for scale in (1, -.75):
                source_value = value * scale
                output = np.empty((5, 2, 3), np.float32)
                jit.invoke(handle, function.__name__, [source_value], [output])
                np.testing.assert_array_equal(output, function(source_value))
                outputs.append(output)
            assert jit.invocation_count() == before + 2
            assert jit.compile_count() == compile_count
            np.testing.assert_array_equal(outputs[0], value.transpose(2, 0, 1))
        finally:
            jit.destroy(handle)


def test_dot_t_ast_reverses_all_result_dimensions():
    from tessera.compiler.graph_ir import GraphIRBuilder
    builder = GraphIRBuilder()
    function = builder.lower(reversed_axis_source)
    assert function.result_types[0].shape == ("5", "3", "2")


@pytest.mark.parametrize("attributes", [
    {"axes": (0, 0, 2)}, {"axes": (0, 1)}, {"axes": (0, 1, 3)},
    {"axes": (True, 1, 2)}, {"axes": (2, 0, 1), "permutation": (0, 1, 2)},
])
def test_frontend_transpose_rejects_invalid_or_conflicting_axes(attributes):
    from tessera.compiler.graph_ir import _shape_transpose, tensor_ir_type
    with pytest.raises(ValueError, match="transpose axes"):
        _shape_transpose([tensor_ir_type((2, 3, 5), "fp32")], attributes)


def test_frontend_empty_permutation_has_canonical_rank_zero_attribute():
    from tessera.compiler.graph_ir import IROp, tensor_ir_type
    scalar = tensor_ir_type((), "fp32")
    operation = IROp("y", "tessera.transpose", ["%x"], [str(scalar)], str(scalar),
                     kwargs={"axes": ()}, inferred_type=scalar)
    assert "permutation = array<i64>" in operation.to_mlir(canonical=True)


@pytest.mark.parametrize(("axes", "expected"), [
    ("permutation = array<i64: 2, 0, 1>,", ["K", "M", "N"]),
    ("", ["K", "N", "M"]),
])
@pytest.mark.parametrize("correct", [True, False])
def test_symbolic_names_follow_actual_axes_even_for_equal_dimensions(opt, axes, expected, correct):
    names = expected if correct else ["M", "N", "K"]
    output_names = ", ".join(f'"{name}"' for name in names)
    source = module("3x3x3xf32", "3x3x3xf32",
                    axes + 'tessera.dim_names_in = ["M", "N", "K"], '
                    + "tessera.dim_names_out = [" + output_names + "]")
    compiled = invoke(opt, source, "--tessera-symdim-equality")
    assert (compiled.returncode == 0) == correct, compiled.stderr
    if not correct:
        assert "SYMDIM_TRANSPOSE_VIOLATION" in compiled.stderr


@pytest.mark.parametrize("correct", [True, False])
def test_symbolic_names_propagate_through_unannotated_transpose(opt, correct):
    names = '["K", "M", "N"]' if correct else '["M", "N", "K"]'
    source = (
        'module { func.func @flow(%x: tensor<3x3x3xf32> '
        '{tessera.dim_names = ["M", "N", "K"]}) -> tensor<3x3x3xf32> {'
        '%a = "tessera.transpose"(%x) {permutation = array<i64: 2, 0, 1>}'
        ' : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>'
        '%b = "tessera.transpose"(%a) {permutation = array<i64: 1, 2, 0>, '
        'tessera.dim_names_in = ' + names + ', '
        'tessera.dim_names_out = ["M", "N", "K"]}'
        ' : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>'
        'return %b : tensor<3x3x3xf32> } }'
    )
    compiled = invoke(opt, source, "--tessera-symdim-equality")
    assert (compiled.returncode == 0) == correct, compiled.stderr
    if not correct:
        assert "SYMDIM_FLOW_INCONSISTENCY" in compiled.stderr


def test_reverse_ad_swaps_symbolic_input_and_output_annotations(opt):
    source = differentiated_source("reverse").replace(
        "permutation = array<i64: 2, 0, 1>",
        'permutation = array<i64: 2, 0, 1>, '
        'tessera.dim_names_in = ["M", "N", "K"], '
        'tessera.dim_names_out = ["K", "M", "N"]')
    differentiated = invoke(opt, source, "--tessera-autodiff-paired",
                            "--tessera-symdim-equality")
    assert differentiated.returncode == 0, differentiated.stderr
    assert 'tessera.dim_names_in = ["K", "M", "N"]' in differentiated.stdout
    assert 'tessera.dim_names_out = ["M", "N", "K"]' in differentiated.stdout


@pytest.mark.parametrize(("source_type", "result_type", "attrs", "expected"), [
    ("2x3x5xf32", "5x2x3xf32", "permutation = array<i64: 2, 0, 1>", [5, 2, 3]),
    ("2x3x5xf32", "5x3x2xf32", "", [5, 3, 2]),
    ("2x3x4x5xf32", "3x5x2x4xf32", "permutation = array<i64: 1, 3, 0, 2>", [3, 5, 2, 4]),
])
def test_native_shape_inference_uses_the_verified_axis_contract(opt, source_type, result_type, attrs, expected):
    import re
    compiled = invoke(opt, module(source_type, result_type, attrs), "--tessera-shape-inference")
    assert compiled.returncode == 0, compiled.stderr
    shape = re.search(r"tessera.inferred_shape = \[([^\]]*)\]", compiled.stdout)
    assert shape is not None, compiled.stdout
    dimensions = [int(part.split(":")[0].strip()) for part in shape.group(1).split(",")]
    assert dimensions == expected
