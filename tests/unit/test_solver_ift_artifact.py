from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.implicit_solver import (
    build_general_solver_product_contract,
    build_physical_general_solver_contract,
    build_solver_ift_contract,
    compile_physical_general_solver_from_graph,
    diagonal_sqrt_ift_reference,
    lower_scheduled_solver_ift,
    package_physical_general_solver,
)
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def test_solver_ift_contract_is_content_addressed_and_fail_closed() -> None:
    first = build_solver_ift_contract(target="x86", shape=(64,))
    second = build_solver_ift_contract(target="x86", shape=(64,))
    rocm = build_solver_ift_contract(target="rocm_gfx1151", shape=(64,))
    assert first == second
    assert first["artifact_hash"] != rocm["artifact_hash"]
    assert first["linear_solve"]["materializes_matrix"] is False
    with pytest.raises(ValueError, match="fail closed"):
        build_solver_ift_contract(target="rocm_gfx1200", shape=(64,))


def test_diagonal_sqrt_reference_exposes_all_ift_phases() -> None:
    theta = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    solution = np.sqrt(theta)
    cotangent = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    residual, linear_solution, parameter_cotangent = diagonal_sqrt_ift_reference(theta, solution, cotangent)
    np.testing.assert_array_equal(residual, np.zeros_like(theta))
    np.testing.assert_allclose(linear_solution, cotangent / (2.0 * solution))
    np.testing.assert_array_equal(parameter_cotangent, linear_solution)


def test_solver_forward_product_has_distinct_nontransposed_lineage() -> None:
    vjp = build_solver_ift_contract(target="x86", shape=(17,))
    jvp = build_solver_ift_contract(target="x86", shape=(17,), product_mode="jvp")
    assert jvp["product_mode"] == "jvp"
    assert jvp["linear_solve"]["transpose"] is False
    assert jvp["artifact_hash"] != vjp["artifact_hash"]
    with pytest.raises(ValueError, match="product_mode"):
        build_solver_ift_contract(target="x86", shape=(17,), product_mode="hvp")


def test_general_solver_product_binds_exact_residual_products_without_promotion() -> None:
    contract = build_general_solver_product_contract(
        target="rocm_gfx1151",
        shape=(17,),
        residual_graph_ir="func.func @residual(%x: tensor<17xf32>)",
        residual_jvp_ir="func.func @residual__jvp(%x: tensor<17xf32>)",
        residual_vjp_ir="func.func @residual__vjp(%x: tensor<17xf32>)",
        product_mode="jvp",
        linear_solver="gmres",
    )
    assert contract["architecture"] == "gfx1151"
    assert contract["matrix_free"] is True
    assert contract["transpose"] is False
    assert contract["execution_state"] == "artifact_only"
    assert len({
        contract["residual_graph_digest"],
        contract["residual_jvp_digest"],
        contract["residual_vjp_digest"],
    }) == 3
    with pytest.raises(ValueError, match="gmres or cg"):
        build_general_solver_product_contract(
            target="x86", shape=(17,), residual_graph_ir="r",
            residual_jvp_ir="j", residual_vjp_ir="v", product_mode="jvp",
            linear_solver="dense",
        )


def _affine_general_children(target: str) -> dict[str, dict]:
    runtime_target = "x86" if target == "x86" else "rocm"

    def binary(kind: str, bindings: list[str]) -> dict:
        return {
            "metadata": {
                "target": runtime_target,
                "compiler_path": f"{runtime_target}_binary_compiled",
                "executable": True,
                "execution_kind": (
                    "native_cpu" if runtime_target == "x86" else "native_gpu"
                ),
                "arg_names": ["a", "b"],
                "ops": [{
                    "op_name": f"tessera.{kind}", "result": "out",
                    "operands": ["a", "b"], "kwargs": {},
                }],
            },
            "bindings": bindings,
        }

    return {
        "residual": binary("sub", ["solution", "parameter"]),
        "solution_jvp": binary("add", ["vector", "zero"]),
        "solution_vjp": binary("add", ["vector", "zero"]),
        "parameter_jvp": binary("mul", ["vector", "minus_one"]),
        "parameter_vjp": binary("mul", ["vector", "minus_one"]),
    }


def _nonlinear_residual_graph(shape: tuple[int, ...]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    text = str(ty)
    return GraphIRModule(functions=[GraphIRFunction(
        name="nonlinear_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)],
        result_types=[ty],
        body=[
            IROp("xx", "tessera.mul", ["%x", "%x"], [text, text], text),
            IROp("sx", "tessera.sin", ["%x"], [text], text),
            IROp("sum", "tessera.add", ["%xx", "%sx"], [text, text], text),
            IROp("residual", "tessera.sub", ["%sum", "%theta"], [text, text], text),
        ],
        return_values=["%residual"],
    )])


def test_residual_graph_compiler_generates_five_distinct_children() -> None:
    package = compile_physical_general_solver_from_graph(
        _nonlinear_residual_graph((7,)), target="x86", shape=(7,),
        tolerance=1.0e-7, max_iterations=16, restart=8,
    )
    children = package.contract["children"]
    assert set(children) == {
        "residual", "solution_jvp", "solution_vjp", "parameter_jvp",
        "parameter_vjp",
    }
    assert len({record["digest"] for record in children.values()}) == 5
    for role, record in children.items():
        program = record["metadata"]["solver_residual_program"]
        assert program["role"] == role
        assert len(program["program_digest"]) == 64
        assert record["metadata"]["compiler_path"] == "x86_solver_graph_compiled"
    assert package.contract["residual_graph_digest"] != package.contract["residual_jvp_digest"]


def test_residual_graph_compiler_fails_closed_for_unsupported_operation() -> None:
    shape = (7,)
    ty = IRType("tensor<7xf32>")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="unsupported_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[IROp("r", "tessera.relu", ["%x"], [str(ty)], str(ty))],
        return_values=["%r"],
    )])
    with pytest.raises(ValueError, match="supports typed pointwise"):
        compile_physical_general_solver_from_graph(
            module, target="x86", shape=shape,
        )


def _graph_type(shape: str, dtype: str = "f32") -> IRType:
    return IRType(f"tensor<{shape}x{dtype}>")


def test_residual_graph_compiler_carries_reduction_and_matmul_products() -> None:
    ty = _graph_type("2x2")
    text = str(ty)
    reduction = GraphIRModule(functions=[GraphIRFunction(
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
    matmul = GraphIRModule(functions=[GraphIRFunction(
        name="matmul_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("xx", "tessera.matmul", ["%x", "%x"], [text, text], text),
            IROp("r", "tessera.sub", ["%xx", "%theta"], [text, text], text),
        ], return_values=["%r"],
    )])
    reduction_package = compile_physical_general_solver_from_graph(
        reduction, target="x86", shape=(2, 2)
    )
    matmul_package = compile_physical_general_solver_from_graph(
        matmul, target="x86", shape=(2, 2)
    )
    reduction_ops = {
        step.get("op_name")
        for step in reduction_package.contract["children"]["solution_vjp"]
        ["metadata"]["solver_residual_program"]["steps"]
    }
    matmul_ops = {
        step.get("op_name")
        for step in matmul_package.contract["children"]["solution_vjp"]
        ["metadata"]["solver_residual_program"]["steps"]
    }
    assert {"tessera.mean", "tessera.broadcast_like", "tessera.mean_adjoint"} <= reduction_ops
    assert {"tessera.matmul", "tessera.transpose"} <= matmul_ops


def test_residual_graph_compiler_unrolls_bounded_region_into_child_lineage(monkeypatch) -> None:
    ty = _graph_type("4")
    text = str(ty)
    body = [IROp("next", "tessera.add", ["%carry", "%x"], [text, text], text)]
    module = GraphIRModule(functions=[GraphIRFunction(
        name="region_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("loop", "tessera.control_for", ["%x"], [text], text,
                 kwargs={"_region": "for", "_trip": 2, "_carry_ssa": "carry",
                         "_next_ssa": "next", "_body": body}),
            IROp("r", "tessera.sub", ["%loop", "%theta"], [text, text], text),
        ], return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(4,)
    )
    program = package.contract["children"]["residual"]["metadata"]["solver_residual_program"]
    assert sum(step.get("op_name") == "tessera.add" for step in program["steps"]) == 2
    assert all(not str(step.get("op_name", "")).startswith("tessera.control_") for step in program["steps"])
    x = np.arange(4, dtype=np.float32)
    theta = 3.0 * x
    vector = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    jvp = _run_generated_child(monkeypatch, package, "solution_jvp", theta, x, vector)
    vjp = _run_generated_child(monkeypatch, package, "solution_vjp", theta, x, vector)
    np.testing.assert_allclose(jvp, 3.0 * vector)
    np.testing.assert_allclose(vjp, 3.0 * vector)


def _conditional_residual_graph() -> GraphIRModule:
    ty = _graph_type("1")
    pred_ty = IRType("tensor<1xi1>")
    text = str(ty)
    return GraphIRModule(functions=[GraphIRFunction(
        name="conditional_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("pred", "tessera.greater", ["%x", "%theta"],
                 [text, text], str(pred_ty)),
            IROp(
                "branch", "tessera.control_if", ["%pred"], [str(pred_ty)], text,
                kwargs={
                    "_region": "if", "_flag_ssa": "pred",
                    "_then_body": [
                        IROp("then", "tessera.mul", ["%x", "%x"],
                             [text, text], text)
                    ],
                    "_then_ssa": "then",
                    "_else_body": [
                        IROp("else", "tessera.add", ["%x", "%theta"],
                             [text, text], text)
                    ],
                    "_else_ssa": "else",
                },
            ),
            IROp("r", "tessera.sub", ["%branch", "%theta"],
                 [text, text], text),
        ], return_values=["%r"],
    )])


def _while_residual_graph() -> GraphIRModule:
    ty = _graph_type("1")
    pred_ty = IRType("tensor<1xi1>")
    text = str(ty)
    return GraphIRModule(functions=[GraphIRFunction(
        name="while_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp(
                "loop", "tessera.control_while", ["%x"], [text], text,
                kwargs={
                    "_region": "while", "_max_iters": 6,
                    "_carry_ssa": "carry",
                    "_cond": [
                        IROp("pred", "tessera.less", ["%carry", "%theta"],
                             [text, text], str(pred_ty))
                    ],
                    "_pred_ssa": "pred",
                    "_body": [
                        IROp("next", "tessera.add", ["%carry", "%x"],
                             [text, text], text)
                    ],
                    "_next_ssa": "next",
                },
            ),
            IROp("r", "tessera.sub", ["%loop", "%theta"],
                 [text, text], text),
        ], return_values=["%r"],
    )])


@pytest.mark.parametrize("module", [_conditional_residual_graph(), _while_residual_graph()])
def test_predicate_regions_have_digest_bound_replay(module: GraphIRModule) -> None:
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(1,)
    )
    for child in package.contract["children"].values():
        program = child["metadata"]["solver_residual_program"]
        replay = program["predicate_replay"]
        assert replay["policy"] == "pure_recompute"
        assert replay["derivative_boundary"] == "reject_equal"
        assert replay["selection_count"] > 0
        assert any(step.get("op_name") == "tessera.where" for step in program["steps"])
        comparisons = [
            step for step in program["steps"]
            if step.get("op_name") == "tessera.greater"
            or step.get("op_name") == "tessera.less"
        ]
        assert comparisons
        assert all(
            step["kwargs"]["derivative_boundary"] == "reject_equal"
            for step in comparisons
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
@pytest.mark.parametrize(
    ("module", "theta", "solution", "expected"),
    [
        (_conditional_residual_graph(), 2.0, 3.0, 6.0),
        (_conditional_residual_graph(), 3.0, 2.0, 1.0),
        (_while_residual_graph(), 4.5, 1.0, 5.0),
    ],
)
def test_predicate_region_products_execute_on_native_target(
    target: str, module: GraphIRModule, theta: float, solution: float,
    expected: float,
) -> None:
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and (
        rt._tessera_opt_path() is None or rt._rocm_chip() != "gfx1151"
    ):
        pytest.skip("exact gfx1151 compiler/device required")
    package = compile_physical_general_solver_from_graph(
        module, target=target, shape=(1,)
    )
    vector = np.array([1.0], dtype=np.float32)
    values = (
        np.array([theta], dtype=np.float32),
        np.array([solution], dtype=np.float32),
        vector,
    )
    for role in ("solution_jvp", "solution_vjp"):
        metadata = package.contract["children"][role]["metadata"]
        result = rt.launch(rt.RuntimeArtifact(metadata=metadata), values)
        assert result["ok"], result.get("reason")
        np.testing.assert_allclose(result["output"], expected, atol=1.0e-6)


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_predicate_region_product_rejects_nondifferentiable_boundary(
    target: str,
) -> None:
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and (
        rt._tessera_opt_path() is None or rt._rocm_chip() != "gfx1151"
    ):
        pytest.skip("exact gfx1151 compiler/device required")
    package = compile_physical_general_solver_from_graph(
        _conditional_residual_graph(), target=target, shape=(1,)
    )
    child = package.contract["children"]["solution_vjp"]["metadata"]
    value = np.array([2.0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=child), (value, value, np.ones_like(value))
    )
    assert result["ok"] is False
    assert "undefined at an equality boundary" in result["reason"]


def test_residual_graph_compiler_binds_dynamic_shape_and_mixed_storage() -> None:
    theta_ty = _graph_type("?x?", "f16")
    x_ty = _graph_type("?x?", "f32")
    out_ty = _graph_type("?x?", "f32")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="dynamic_mixed_residual",
        args=[IRArg("theta", theta_ty), IRArg("x", x_ty)], result_types=[out_ty],
        body=[IROp("r", "tessera.sub", ["%x", "%theta"],
                   [str(x_ty), str(theta_ty)], str(out_ty))],
        return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(8, 8)
    )
    contract = package.contract["value_contract"]
    assert contract["shape_bounds"] == [8, 8]
    assert contract["dynamic_dimensions"] == [0, 1]
    assert contract["storage_dtypes"] == {
        "parameter": "f16", "solution": "f32", "residual": "f32",
        "product": "f32",
    }
    assert contract["accumulation_dtype"] == "f32"

    from tessera import runtime as rt
    artifact = rt.RuntimeArtifact(metadata=package.runtime_metadata())
    with pytest.raises(ValueError, match="exceeds its bounded-dynamic contract"):
        rt._execute_x86_physical_general_solver(
            artifact,
            (np.zeros((9, 9), np.float16), np.zeros((9, 9), np.float32),
             np.zeros((9, 9), np.float32)),
        )
    with pytest.raises(ValueError, match="parameter storage dtype"):
        rt._execute_x86_physical_general_solver(
            artifact,
            (np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32),
             np.zeros((4, 4), np.float32)),
        )


def _install_numpy_solver_lanes(monkeypatch):
    from tessera import runtime as rt

    def binary(artifact, operands):
        op = artifact.metadata["ops"][0]["op_name"]
        lhs, rhs = operands
        if op == "tessera.add":
            return lhs + rhs
        if op == "tessera.sub":
            return lhs - rhs
        if op == "tessera.mul":
            return lhs * rhs
        if op == "tessera.div":
            return lhs / rhs
        raise AssertionError(f"unexpected binary op {op}")

    def reduce(artifact, operands):
        op = artifact.metadata["ops"][0]
        fn = np.mean if op["op_name"] == "tessera.mean" else np.sum
        return fn(operands[0], **op["kwargs"])

    monkeypatch.setattr(rt, "_execute_x86_compiled_binary", binary)
    monkeypatch.setattr(rt, "_execute_x86_compiled_reduce", reduce)
    monkeypatch.setattr(rt, "_x86_gemm_2d", lambda lhs, rhs: lhs @ rhs)


def _run_generated_child(monkeypatch, package, role: str, *args):
    from tessera import runtime as rt

    _install_numpy_solver_lanes(monkeypatch)
    metadata = package.contract["children"][role]["metadata"]
    return rt._execute_solver_residual_program(
        rt.RuntimeArtifact(metadata=metadata), args, target="x86"
    )


def test_bounded_dynamic_mixed_storage_executes_full_matrix_free_parent(monkeypatch) -> None:
    from tessera import runtime as rt

    theta_ty = _graph_type("?x?", "f16")
    x_ty = _graph_type("?x?", "f32")
    out_ty = _graph_type("?x?", "f32")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="dynamic_mixed_residual",
        args=[IRArg("theta", theta_ty), IRArg("x", x_ty)], result_types=[out_ty],
        body=[IROp("r", "tessera.sub", ["%x", "%theta"],
                   [str(x_ty), str(theta_ty)], str(out_ty))],
        return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(8, 8), tolerance=1.0e-7,
        max_iterations=8, restart=4,
    )
    _install_numpy_solver_lanes(monkeypatch)

    def child_launch(artifact, operands):
        return {
            "ok": True,
            "output": rt._execute_solver_residual_program(
                artifact, operands, target="x86"
            ),
        }

    monkeypatch.setattr(rt, "launch", child_launch)
    theta = np.linspace(0.5, 2.0, 15, dtype=np.float16).reshape(3, 5)
    solution = theta.astype(np.float32)
    cotangent = np.linspace(-1.0, 1.0, 15, dtype=np.float32).reshape(3, 5)
    residual, linear, parameter_cotangent = rt._execute_x86_physical_general_solver(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (theta, solution, cotangent),
    )
    np.testing.assert_array_equal(residual, np.zeros_like(solution))
    np.testing.assert_allclose(linear, cotangent, atol=1.0e-7)
    np.testing.assert_allclose(parameter_cotangent, cotangent, atol=1.0e-7)


def test_generated_reduction_products_execute_with_broadcast_duality(monkeypatch) -> None:
    ty = _graph_type("2x2")
    text = str(ty)
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
        module, target="x86", shape=(2, 2)
    )
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    theta = x + np.mean(x, axis=1, keepdims=True)
    vector = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)
    jvp = _run_generated_child(monkeypatch, package, "solution_jvp", theta, x, vector)
    vjp = _run_generated_child(monkeypatch, package, "solution_vjp", theta, x, vector)
    np.testing.assert_allclose(jvp, vector + np.mean(vector, axis=1, keepdims=True))
    np.testing.assert_allclose(vjp, vector + np.sum(vector, axis=1, keepdims=True) / 2.0)


def test_generated_matmul_products_execute_exact_jvp_and_vjp(monkeypatch) -> None:
    ty = _graph_type("2x2")
    text = str(ty)
    module = GraphIRModule(functions=[GraphIRFunction(
        name="matmul_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("xx", "tessera.matmul", ["%x", "%x"], [text, text], text),
            IROp("r", "tessera.sub", ["%xx", "%theta"], [text, text], text),
        ], return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(2, 2)
    )
    x = np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32)
    theta = x @ x
    vector = np.array([[1.0, -2.0], [0.25, 4.0]], dtype=np.float32)
    jvp = _run_generated_child(monkeypatch, package, "solution_jvp", theta, x, vector)
    vjp = _run_generated_child(monkeypatch, package, "solution_vjp", theta, x, vector)
    np.testing.assert_allclose(jvp, vector @ x + x @ vector)
    np.testing.assert_allclose(vjp, vector @ x.T + x.T @ vector)


def test_generated_matmul_allows_distinct_parameter_and_solution_spaces(monkeypatch) -> None:
    theta_ty = _graph_type("3x3")
    x_ty = _graph_type("2x3")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="parameterized_matmul_residual",
        args=[IRArg("theta", theta_ty), IRArg("x", x_ty)], result_types=[x_ty],
        body=[
            IROp("xt", "tessera.matmul", ["%x", "%theta"],
                 [str(x_ty), str(theta_ty)], str(x_ty)),
            IROp("r", "tessera.sub", ["%xt", "%x"],
                 [str(x_ty), str(x_ty)], str(x_ty)),
        ], return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="x86", shape=(2, 3)
    )
    assert package.contract["value_contract"]["parameter_shape_bounds"] == [3, 3]
    theta = np.array(
        [[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [1.0, 0.5, 2.0]],
        dtype=np.float32,
    )
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    solution_tangent = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3)
    parameter_tangent = np.eye(3, dtype=np.float32) * 0.25
    cotangent = np.linspace(0.5, 2.0, 6, dtype=np.float32).reshape(2, 3)
    solution_jvp = _run_generated_child(
        monkeypatch, package, "solution_jvp", theta, x, solution_tangent
    )
    parameter_jvp = _run_generated_child(
        monkeypatch, package, "parameter_jvp", theta, x, parameter_tangent
    )
    parameter_vjp = _run_generated_child(
        monkeypatch, package, "parameter_vjp", theta, x, cotangent
    )
    np.testing.assert_allclose(solution_jvp, solution_tangent @ theta - solution_tangent)
    np.testing.assert_allclose(parameter_jvp, x @ parameter_tangent)
    np.testing.assert_allclose(parameter_vjp, x.T @ cotangent)


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_generated_nonlinear_residual_children_execute_exact_products(target: str) -> None:
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and (
        rt._tessera_opt_path() is None or rt._rocm_chip() != "gfx1151"
    ):
        pytest.skip("exact gfx1151 compiler/device required")
    shape = (7,)
    package = compile_physical_general_solver_from_graph(
        _nonlinear_residual_graph(shape), target=target, shape=shape,
        tolerance=2.0e-6, max_iterations=16, restart=8,
    )
    solution = np.linspace(0.75, 1.75, shape[0], dtype=np.float32)
    parameter = (solution * solution + np.sin(solution)).astype(np.float32)
    cotangent = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, cotangent),
    )
    assert result["ok"], result.get("reason")
    residual, linear, parameter_cotangent = result["output"]
    expected = cotangent / (2.0 * solution + np.cos(solution))
    np.testing.assert_allclose(residual, 0.0, atol=3.0e-6)
    np.testing.assert_allclose(linear, expected, rtol=2.0e-5, atol=3.0e-6)
    np.testing.assert_allclose(
        parameter_cotangent, expected, rtol=2.0e-5, atol=3.0e-6
    )


def test_physical_general_solver_contract_binds_all_native_children() -> None:
    children = _affine_general_children("x86")
    contract = build_physical_general_solver_contract(
        target="x86", shape=(17,), residual_graph_ir="R=x-theta",
        residual_jvp_ir="dR=dx-dtheta", residual_vjp_ir="R^T products",
        product_mode="vjp", children=children,
    )
    assert contract["execution_state"] == "native_composite"
    assert contract["linear_solver"] == "gmres"
    assert set(contract["children"]) == {
        "residual", "solution_jvp", "solution_vjp", "parameter_jvp",
        "parameter_vjp",
    }
    assert len({child["digest"] for child in contract["children"].values()}) >= 3
    stale = dict(children)
    stale.pop("parameter_vjp")
    with pytest.raises(ValueError, match="requires residual"):
        build_physical_general_solver_contract(
            target="x86", shape=(17,), residual_graph_ir="r",
            residual_jvp_ir="j", residual_vjp_ir="v", product_mode="vjp",
            children=stale,
        )


def test_general_solver_never_substitutes_gmres_for_declared_cg() -> None:
    from tessera import runtime as rt

    package = package_physical_general_solver(
        target="x86", shape=(4,), residual_graph_ir="R=x-theta",
        residual_jvp_ir="dR=dx-dtheta", residual_vjp_ir="R^T products",
        product_mode="vjp", children=_affine_general_children("x86"),
        linear_solver="cg", tolerance=1.0e-6, max_iterations=8, restart=4,
    )
    value = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="CG requires the target-owned SPD"):
        rt._execute_x86_physical_general_solver(
            rt.RuntimeArtifact(metadata=package.runtime_metadata()),
            (value, value, value),
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_physical_general_solver_executes_native_matrix_free_children(target: str) -> None:
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and (
        rt._tessera_opt_path() is None or rt._rocm_chip() != "gfx1151"
    ):
        pytest.skip("exact gfx1151 compiler/device required")
    shape = (19,)
    package = package_physical_general_solver(
        target=target, shape=shape, residual_graph_ir="R=x-theta",
        residual_jvp_ir="dR=dx-dtheta", residual_vjp_ir="R^T products",
        product_mode="vjp", children=_affine_general_children(target),
        tolerance=1.0e-7, max_iterations=16, restart=8,
    )
    rng = np.random.default_rng(631)
    parameter = rng.normal(size=shape).astype(np.float32)
    solution = parameter.copy()
    cotangent = rng.normal(size=shape).astype(np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, cotangent),
    )
    assert result["ok"], result.get("reason")
    residual, linear, parameter_cotangent = result["output"]
    np.testing.assert_allclose(residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(linear, cotangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        parameter_cotangent, cotangent, rtol=1e-6, atol=1e-6
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151", "nvidia_sm120"])
def test_solver_forward_product_consumes_residual_jvp(target: str) -> None:
    artifact = lower_scheduled_solver_ift(
        target=target, shape=(17,), product_mode="jvp"
    )
    artifact.validate()
    assert "tessera_solver.residual_jvp" in artifact.shared_solver_ir
    assert 'product_mode = "jvp"' in artifact.schedule_ir
    assert "transpose = false" in artifact.schedule_ir


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151", "nvidia_sm120"])
def test_solver_forward_product_executes_parameter_tangent(target: str) -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("production x86 AVX-512 image unavailable")
    if target == "rocm_gfx1151" and (
        rt._rocm_chip() != "gfx1151" or not rt._rocm_wmma_runtime_available()
    ):
        pytest.skip("exact gfx1151 runtime unavailable")
    if target == "nvidia_sm120":
        pytest.skip("exact SM120 execution lives in the NVIDIA device packet")
    parameter = np.linspace(0.5, 8.5, 257, dtype=np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    dparameter = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    artifact = lower_scheduled_solver_ift(
        target=target, shape=parameter.shape, product_mode="jvp"
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=artifact.runtime_metadata()),
        (parameter, solution, dparameter),
    )
    assert result["ok"], result.get("reason")
    residual, solution_tangent, parameter_product = result["output"]
    np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    expected = dparameter / (2.0 * solution)
    np.testing.assert_allclose(solution_tangent, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(parameter_product, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151", "nvidia_sm120"])
def test_solver_ift_consumes_shared_chain_into_one_tile_artifact(target: str) -> None:
    artifact = lower_scheduled_solver_ift(target=target, shape=(64,))
    artifact.validate()
    assert artifact.shared_solver_ir.count("tessera_solver.linear_solve") == 1
    assert artifact.tile_ir.count("tile.solver_ift_kernel") == 1
    assert artifact.artifact_hash in artifact.tile_ir


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_solver_ift_compiled_package_matches_numerical_oracle(target: str) -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("production x86 AVX-512 image unavailable")
    if target == "rocm_gfx1151" and not rt._rocm_wmma_runtime_available():
        pytest.skip("gfx1151 HIP runtime unavailable")

    rng = np.random.default_rng(20260809)
    parameter = rng.uniform(0.25, 16.0, size=(3, 257)).astype(np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    cotangent = rng.standard_normal(parameter.shape).astype(np.float32)
    scheduled = lower_scheduled_solver_ift(target=target, shape=parameter.shape)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=scheduled.runtime_metadata()),
        (parameter, solution, cotangent),
    )
    assert result["ok"] is True, result.get("reason")
    expected = diagonal_sqrt_ift_reference(parameter, solution, cotangent)
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_solver_ift_tile_artifacts_reach_architecture_owned_lowering() -> None:
    tool = find_tessera_opt()
    assert tool is not None
    x86 = lower_scheduled_solver_ift(target="x86", shape=(17,))
    x86_ir = run_tessera_opt(
        tool,
        x86.tile_ir,
        "--pass-pipeline=builtin.module(tessera-x86-executable{"
        "family=solver_ift input=tile output=target arch=x86_64_avx512})",
    )
    assert "tessera_x86_avx512_solver_ift_sqrt_f32" in x86_ir
    assert "tessera_x86.abi_call" in x86_ir
    assert "tile.solver_ift_kernel" not in x86_ir

    rocm = lower_scheduled_solver_ift(target="rocm_gfx1151", shape=(17,))
    target_ir = run_tessera_opt(tool, rocm.tile_ir, "--lower-tile-to-rocm=arch=gfx1151")
    assert target_ir.count("tessera_rocm.solver_ift") == 1
    gpu_ir = run_tessera_opt(tool, target_ir, "--generate-rocm-solver-ift-kernel")
    assert "gpu.func @tessera_solver_ift" in gpu_ir
    assert "tessera_rocm.solver_ift" not in gpu_ir


def test_solver_ift_rocm_runtime_fails_closed_off_gfx1151(monkeypatch) -> None:
    from tessera import runtime as rt

    shape = (4,)
    contract = build_solver_ift_contract(target="rocm_gfx1151", shape=shape)
    metadata = {
        "target": "rocm",
        "compiler_path": "rocm_solver_ift_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["parameter", "solution", "cotangent"],
        "scheduled_solver_ift": contract,
    }
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1200")
    value = np.ones(shape, dtype=np.float32)
    result = rt.launch(rt.RuntimeArtifact(metadata=metadata), (value, value, value))
    assert result["ok"] is False
    assert "verified for gfx1151" in result["reason"]
