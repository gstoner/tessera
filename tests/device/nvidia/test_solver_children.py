"""Exact-SM120 certificates for typed solver residual children."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.implicit_solver import compile_physical_general_solver_from_graph
from tests._support.nvidia import assert_native_gpu, require_nvidia_mma_runtime


pytestmark = [pytest.mark.slow, pytest.mark.hardware_nvidia]


def _storages(values: np.ndarray):
    yield "f32", values.astype(np.float32)
    yield "f16", values.astype(np.float16)
    try:
        import ml_dtypes
        yield "bf16", values.astype(ml_dtypes.bfloat16)
    except ImportError:
        pass


def _runtime_artifact(path: str, op_name: str, operands: list[str], kwargs=None):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": path,
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands,
        "ops": [{"op_name": op_name, "result": "out",
                 "operands": operands, "kwargs": kwargs or {}}],
    })


def test_registered_solver_child_runtime_paths() -> None:
    rt = require_nvidia_mma_runtime()
    a = np.linspace(0.25, 2.0, 33, dtype=np.float32)
    b = np.linspace(2.0, 0.25, 33, dtype=np.float32)
    unary = rt.launch(
        _runtime_artifact("nvidia_unary_compiled", "tessera.sqrt", ["x"]), (a,)
    )
    assert unary["ok"], unary.get("reason")
    np.testing.assert_allclose(unary["output"], np.sqrt(a), rtol=2e-6, atol=2e-6)
    compare = rt.launch(
        _runtime_artifact("nvidia_compare_compiled", "tessera.less", ["a", "b"]),
        (a, b),
    )
    assert compare["ok"], compare.get("reason")
    select = rt.launch(
        _runtime_artifact("nvidia_where_compiled", "tessera.where", ["p", "a", "b"]),
        (compare["output"], a, b),
    )
    assert select["ok"], select.get("reason")
    np.testing.assert_array_equal(select["output"], np.where(a < b, a, b))


@pytest.mark.parametrize(
    "kind,reference",
    [
        (0, np.sqrt), (1, np.reciprocal), (2, np.exp), (3, np.log),
        (4, np.tanh), (5, lambda x: 1.0 / (1.0 + np.exp(-x))),
        (6, np.sin), (7, np.cos),
    ],
)
def test_solver_unary_dtype_policy(kind, reference) -> None:
    require_nvidia_mma_runtime()
    from tessera.compiler.emit.nvidia_cuda import run_solver_unary

    base = np.linspace(0.25, 2.0, 257, dtype=np.float32)
    for name, value in _storages(base):
        actual = run_solver_unary(value, kind)
        expected = reference(value.astype(np.float32)).astype(value.dtype)
        tolerance = 2e-6 if name == "f32" else (2e-3 if name == "f16" else 2e-2)
        np.testing.assert_allclose(
            actual.astype(np.float32), expected.astype(np.float32),
            rtol=tolerance, atol=tolerance,
        )


@pytest.mark.parametrize("kind", range(6))
def test_solver_comparison_and_where_dtype_policy(kind: int) -> None:
    require_nvidia_mma_runtime()
    from tessera.compiler.emit.nvidia_cuda import run_solver_compare, run_solver_where

    lhs = np.linspace(-2.0, 2.0, 259, dtype=np.float32)
    rhs = np.linspace(1.5, -1.5, 259, dtype=np.float32)
    refs = (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)
    for _name, a in _storages(lhs):
        b = rhs.astype(a.dtype)
        predicate = run_solver_compare(a, b, kind)
        np.testing.assert_array_equal(predicate, refs[kind](a, b))
        actual = run_solver_where(predicate, a, b)
        np.testing.assert_array_equal(actual, np.where(predicate, a, b))


def test_solver_reduction_supports_bf16_storage() -> None:
    require_nvidia_mma_runtime()
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from tessera.compiler.emit.nvidia_cuda import run_row_reduce

    value = np.linspace(-1.0, 2.0, 3 * 257, dtype=np.float32).reshape(3, 257)
    stored = value.astype(ml_dtypes.bfloat16)
    for kind, fn in (("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)):
        actual = run_row_reduce(stored, kind)
        expected = fn(stored.astype(np.float32), axis=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_solver_ieee_matmul_never_uses_tf32() -> None:
    require_nvidia_mma_runtime()
    from tessera.compiler.emit.nvidia_cuda import run_solver_matmul_ieee_f32

    rng = np.random.default_rng(827)
    a = rng.standard_normal((19, 23)).astype(np.float32)
    b = rng.standard_normal((23, 17)).astype(np.float32)
    actual = run_solver_matmul_ieee_f32(a, b)
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_solver_low_precision_matmul_uses_explicit_native_mma_policy(storage: str) -> None:
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(828)
    a_f32 = rng.standard_normal((32, 48), dtype=np.float32)
    b_f32 = rng.standard_normal((48, 16), dtype=np.float32)
    if storage == "f16":
        a, b = a_f32.astype(np.float16), b_f32.astype(np.float16)
    else:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        a, b = a_f32.astype(ml_dtypes.bfloat16), b_f32.astype(ml_dtypes.bfloat16)
    policy = {"storage": storage, "accum": "fp32", "math_mode": "ieee"}
    artifact = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "execution_kind": "native_gpu",
        "executable": True, "compiler_path": "nvidia_solver_matmul_compiled",
        "arg_names": ["a", "b"],
        "ops": [{"op_name": "tessera.matmul", "operands": ["a", "b"],
                 "kwargs": {"numeric_policy": policy}}],
    })
    result = rt.launch(artifact, (a, b))
    assert_native_gpu(result)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(result["output"], expected, rtol=3e-5, atol=3e-5)

    missing_storage = dict(artifact.metadata)
    missing_storage["ops"] = [{
        "op_name": "tessera.matmul", "operands": ["a", "b"],
        "kwargs": {"numeric_policy": {"accum": "fp32", "math_mode": "ieee"}},
    }]
    rejected = rt.launch(rt.RuntimeArtifact(metadata=missing_storage), (a, b))
    assert rejected["ok"] is False
    assert "storage must explicitly match" in rejected["reason"]


def test_compiler_generated_reduction_products_execute_on_cuda() -> None:
    rt = require_nvidia_mma_runtime()
    ty = IRType("tensor<2x4xf32>"); text = str(ty)
    module = GraphIRModule(functions=[GraphIRFunction(
        name="reduction_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("m", "tessera.mean", ["%x"], [text], "tensor<2x1xf32>",
                 kwargs={"axis": 1, "keepdims": True}),
            IROp("xm", "tessera.add", ["%x", "%m"],
                 [text, "tensor<2x1xf32>"], text),
            IROp("r", "tessera.sub", ["%xm", "%theta"], [text, text], text),
        ], return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="nvidia_sm120", shape=(2, 4),
        tolerance=2e-6, max_iterations=12, restart=6,
    )
    solution = np.arange(8, dtype=np.float32).reshape(2, 4) / 4.0
    parameter = solution + solution.mean(axis=1, keepdims=True)
    product = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, product),
    )
    assert result["ok"], result.get("reason")
    expected = product - product.mean(axis=1, keepdims=True) / 2.0
    np.testing.assert_allclose(result["output"][0], 0.0, atol=2e-6)
    np.testing.assert_allclose(result["output"][1], expected, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(result["output"][2], expected, rtol=2e-5, atol=3e-6)


def test_compiler_generated_predicate_where_products_execute_on_cuda() -> None:
    rt = require_nvidia_mma_runtime()
    ty = IRType("tensor<1xf32>"); pred = IRType("tensor<1xi1>"); text = str(ty)
    module = GraphIRModule(functions=[GraphIRFunction(
        name="predicate_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("flag", "tessera.greater", ["%x", "%theta"],
                 [text, text], str(pred)),
            IROp("branch", "tessera.control_if", ["%flag"], [str(pred)], text,
                 kwargs={
                     "_region": "if", "_flag_ssa": "flag",
                     "_then_body": [IROp(
                         "then", "tessera.add", ["%x", "%x"],
                         [text, text], text,
                     )],
                     "_then_ssa": "then",
                     "_else_body": [IROp(
                         "else", "tessera.add", ["%x", "%theta"],
                         [text, text], text,
                     )],
                     "_else_ssa": "else",
                 }),
        ], return_values=["%branch"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="nvidia_sm120", shape=(1,)
    )
    parameter = np.array([0.0], dtype=np.float32)
    solution = np.array([2.0], dtype=np.float32)
    vector = np.ones(1, dtype=np.float32)
    for role in ("solution_jvp", "solution_vjp"):
        child = package.contract["children"][role]["metadata"]
        result = rt.launch(
            rt.RuntimeArtifact(metadata=child),
            (parameter, solution, vector),
        )
        assert result["ok"], result.get("reason")
        np.testing.assert_array_equal(result["output"], [2.0])


def test_compiler_generated_matmul_products_require_ieee_policy() -> None:
    rt = require_nvidia_mma_runtime()
    ty = IRType("tensor<2x2xf32>"); text = str(ty)
    policy = {"storage": "f32", "accumulation": "f32",
              "numeric_policy": {"math_mode": "ieee", "accum": "fp32"}}
    module = GraphIRModule(functions=[GraphIRFunction(
        name="matmul_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("xt", "tessera.matmul", ["%x", "%theta"],
                 [text, text], text, kwargs=policy),
            IROp("r", "tessera.sub", ["%xt", "%x"], [text, text], text),
        ], return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="nvidia_sm120", shape=(2, 2)
    )
    theta = np.array([[2.0, 0.5], [0.25, 3.0]], dtype=np.float32)
    solution = np.arange(4, dtype=np.float32).reshape(2, 2)
    vector = np.array([[1.0, -2.0], [0.25, 4.0]], dtype=np.float32)
    expected = {
        "solution_jvp": vector @ theta - vector,
        "solution_vjp": vector @ theta.T - vector,
    }
    for role, oracle in expected.items():
        child = package.contract["children"][role]["metadata"]
        result = rt.launch(
            rt.RuntimeArtifact(metadata=child), (theta, solution, vector)
        )
        assert result["ok"], result.get("reason")
        np.testing.assert_allclose(result["output"], oracle, rtol=2e-6, atol=2e-6)

    no_policy = dict(policy); no_policy["numeric_policy"] = {}
    bad_module = GraphIRModule(functions=[GraphIRFunction(
        name="matmul_without_policy",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[IROp("r", "tessera.matmul", ["%x", "%theta"],
                   [text, text], text, kwargs=no_policy)],
        return_values=["%r"],
    )])
    bad = compile_physical_general_solver_from_graph(
        bad_module, target="nvidia_sm120", shape=(2, 2)
    )
    child = bad.contract["children"]["residual"]["metadata"]
    result = rt.launch(rt.RuntimeArtifact(metadata=child), (theta, solution))
    assert result["ok"] is False
    assert "requires explicit numeric_policy" in result["reason"]
