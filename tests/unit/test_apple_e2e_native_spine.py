"""APPLE-E2E-1 host-free contracts plus explicitly gated exact-device proofs."""
from __future__ import annotations

import hashlib

import pytest
import numpy as np

from tessera.compiler.apple_native import APPLE_BMM_F32_ABI, APPLE_BMM_F32_SYMBOL
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.pipeline_registry import target_pipeline_lookup


def _module(dtype: str = "fp32") -> GraphIRModule:
    spelling = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[dtype]
    a = IRType(f"tensor<2x4x8x{spelling}>", ("2", "4", "8"), dtype)
    b = IRType(f"tensor<2x8x16x{spelling}>", ("2", "8", "16"), dtype)
    o = IRType(f"tensor<2x4x16x{spelling}>", ("2", "4", "16"), dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="bmm", args=[IRArg("a", a), IRArg("b", b)], result_types=[o],
        body=[IROp(result="out", op_name="tessera.batched_gemm", operands=["%a", "%b"],
                    operand_types=[str(a), str(b)], result_type=str(o))], return_values=["%out"],
    )])


def _softmax_module() -> GraphIRModule:
    x = IRType("tensor<3x7xf32>", ("3", "7"), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="softmax", args=[IRArg("x", x)], result_types=[x],
        body=[IROp(result="out", op_name="tessera.softmax", operands=["%x"],
                    operand_types=[str(x)], result_type=str(x), kwargs={"axis": -1})],
        return_values=["%out"],
    )])


def _transpose_module(dtype: str, shape: tuple[int, ...], axes: tuple[int, ...]) -> GraphIRModule:
    spelling = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[dtype]
    input_type = IRType(
        "tensor<" + "x".join(map(str, shape)) + "x" + spelling + ">",
        tuple(map(str, shape)), dtype,
    )
    output_shape = tuple(shape[axis] for axis in axes)
    output_type = IRType(
        "tensor<" + "x".join(map(str, output_shape)) + "x" + spelling + ">",
        tuple(map(str, output_shape)), dtype,
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="transpose", args=[IRArg("x", input_type)], result_types=[output_type],
        body=[IROp(result="out", op_name="tessera.transpose", operands=["%x"],
                    operand_types=[str(input_type)], result_type=str(output_type),
                    kwargs={"axes": axes})], return_values=["%out"],
    )])


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize(("dtype", "shape", "axes"), [
    (np.float32, (3, 7), (1, 0)),
    (np.float32, (2, 3, 5), (2, 0, 1)),
    (np.float16, (3, 7), (1, 0)),
    (np.float16, (2, 3, 5), (2, 0, 1)),
    (pytest.param("bf16", (3, 7), (1, 0),
                  marks=pytest.mark.skipif(
                      __import__("importlib").util.find_spec("ml_dtypes") is None,
                      reason="ml_dtypes is required for bf16 transpose coverage"))),
    (pytest.param("bf16", (2, 3, 5), (2, 0, 1),
                  marks=pytest.mark.skipif(
                      __import__("importlib").util.find_spec("ml_dtypes") is None,
                      reason="ml_dtypes is required for bf16 transpose coverage"))),
])
def test_native_transpose_execute_compare_on_metal(dtype, shape, axes):
    """The native descriptor carries the permuted output shape and axes ABI."""
    from tessera.compiler.driver import compile_graph_module
    from tessera.runtime import RuntimeArtifact, launch

    storage = __import__("ml_dtypes").bfloat16 if dtype == "bf16" else dtype
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(storage)
    expected = np.transpose(x, axes)
    package_dtype = {np.float32: "fp32", np.float16: "fp16", "bf16": "bf16"}[dtype]
    bundle = compile_graph_module(
        _transpose_module(package_dtype, shape, axes), source_origin="apple-e2e-transpose-descriptor-test",
        target="apple_gpu", options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["route"] == "apple_mpsgraph_transpose_native_library"
    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=bundle.target_ir.text, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
    )
    for _ in range(3):
        actual = launch(artifact, {"x": x, "out": np.empty(expected.shape, dtype=storage)})
        assert actual["ok"], actual
        assert actual["execution_kind"] == "native_gpu"
        actual = actual["output"]
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_apple_gpu_canonical_pipeline_is_the_runtime_pipeline() -> None:
    resolution = target_pipeline_lookup("apple_gpu")
    assert resolution is not None
    assert resolution.current_driver_pipeline == "tessera-lower-to-apple_gpu-runtime"
    assert resolution.declared_pipeline == "tessera-lower-to-apple_gpu-runtime"


def test_canonical_compile_promotes_only_descriptor_contracts(monkeypatch, tmp_path) -> None:
    """The normal canonical path owns the descriptor; explicit value mode does not.

    This preserves the value-IR probe surface while ensuring APPLE-E2E-1's
    static packaged scope reaches the generic descriptor launcher by default.
    """
    from tessera.compiler.driver import canonical_compile_options

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-e2e-test-runtime")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(dylib))
    assert canonical_compile_options(_module(), target="apple_gpu")["package_native"] is True
    assert canonical_compile_options(
        _module(), target="apple_gpu", options={"apple_target_ir_mode": "value"},
    )["package_native"] is False


def test_value_family_descriptor_states_are_explicit() -> None:
    from tessera.compiler.apple_native import value_descriptor_state

    assert value_descriptor_state("tessera.rl.ppo_policy_loss") == "descriptor_ready"
    assert value_descriptor_state("tessera.ebm.energy_quadratic") == "descriptor_ready"
    assert value_descriptor_state("tessera.ebm.langevin_step") == "descriptor_ready"
    assert value_descriptor_state("tessera.ebm.refinement") == "descriptor_ready"
    assert value_descriptor_state("tessera.ebm.partition_exact") == "descriptor_ready"
    assert value_descriptor_state("tessera.clifford.geometric_product") == "descriptor_ready"
    assert value_descriptor_state("tessera.cholesky") == "descriptor_ready"
    assert value_descriptor_state("tessera.cholesky_solve") == "descriptor_ready"
    assert value_descriptor_state("tessera.transpose") == "descriptor_ready"
    assert value_descriptor_state("tessera.svd") == "unsupported"


@pytest.mark.parametrize(("dtype", "symbol", "abi"), [
    ("fp32", "tessera_apple_gpu_bmm_f32", "tessera.apple.bmm.a_b_o_batch_m_n_k.f32.v1"),
    ("fp16", "tessera_apple_gpu_bmm_f16", "tessera.apple.bmm.a_b_o_batch_m_n_k.f16.v1"),
    ("bf16", "tessera_apple_gpu_bmm_bf16", "tessera.apple.bmm.a_b_o_batch_m_n_k.bf16.v1"),
])
def test_apple_native_package_hashes_dylib_and_names_abi(monkeypatch, tmp_path, dtype, symbol, abi) -> None:
    from tessera.compiler import apple_native

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-e2e-test-runtime")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(dylib))
    package = apple_native.package_batched_gemm(
        _module(dtype), pipeline_name="tessera-lower-to-apple_gpu-runtime",
    )
    assert package.image.payload_digest == hashlib.sha256(dylib.read_bytes()).hexdigest()
    assert package.image.entry_points[0].symbol == symbol
    assert package.descriptor.abi_id == abi
    assert package.descriptor.entry_symbol == symbol
    assert package.descriptor.buffers[0].dtype == dtype
    assert package.descriptor.provenance["work_item"] == "APPLE-E2E-1"


def test_apple_softmax_package_hashes_dylib_and_names_abi(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-e2e-test-runtime")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(dylib))
    package = apple_native.package_native(
        _softmax_module(), pipeline_name="tessera-lower-to-apple_gpu-runtime",
    )
    assert package.image.payload_digest == hashlib.sha256(dylib.read_bytes()).hexdigest()
    assert package.image.entry_points[0].symbol == "tessera_apple_gpu_softmax_f32"
    assert package.descriptor.abi_id == "tessera.apple.softmax.x_o_rows_columns.f32.v1"
    assert package.descriptor.provenance["route"] == "apple_softmax_native_library"


def _value_module(op_name, arg_shapes, out_shape, *, kwargs=None):
    args = []
    operands = []
    for index, shape in enumerate(arg_shapes):
        text = "tensor<" + "x".join(map(str, shape)) + "xf32>" if shape else "tensor<f32>"
        args.append(IRArg(f"a{index}", IRType(text, tuple(map(str, shape)), "fp32")))
        operands.append(f"%a{index}")
    out_text = "tensor<" + "x".join(map(str, out_shape)) + "xf32>" if out_shape else "tensor<f32>"
    out_type = IRType(out_text, tuple(map(str, out_shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=args, result_types=[out_type],
        body=[IROp(result="out", op_name=op_name, operands=operands,
                    operand_types=[str(arg.ir_type) for arg in args], result_type=str(out_type), kwargs=kwargs or {})],
        return_values=["%out"],
    )])


def _tuple_value_module(op_name, input_shape, result_specs):
    """One static multi-result Graph-IR call for the CPU descriptor ABI."""
    input_type = IRType(
        "tensor<" + "x".join(map(str, input_shape)) + "xf32>",
        tuple(map(str, input_shape)), "fp32",
    )
    result_types = [
        IRType(
            "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">",
            tuple(map(str, shape)), dtype,
        )
        for _, shape, dtype in result_specs
    ]
    result_names = [name for name, _, _ in result_specs]
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=[IRArg("a0", input_type)], result_types=result_types,
        body=[IROp(result=",".join(result_names), op_name=op_name, operands=["%a0"],
                    operand_types=[str(input_type)],
                    result_type="(" + ", ".join(map(str, result_types)) + ")")],
        return_values=["%" + name for name in result_names],
    )])


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("op,shapes,out_shape,kwargs,reference", [
    ("tessera.rl.ppo_policy_loss", [(2, 3), (2, 3), (2, 3)], (), {"clip_epsilon": 0.2, "reduction": "mean"}, lambda x: -np.mean(np.minimum(np.exp(x[0]-x[1])*x[2], np.clip(np.exp(x[0]-x[1]), .8, 1.2)*x[2]))),
    ("tessera.rl.ppo_policy_loss", [(2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)], (), {"clip_epsilon": 0.2, "reduction": "mean", "kl_coef": 0.03, "entropy_coef": 0.02}, lambda x: _ppo_extended_reference(x, 0.2, 0.03, 0.02)),
    ("tessera.ebm.energy_quadratic", [(2, 3), (2, 3)], (2,), {}, lambda x: .5*np.sum((x[0]-x[1])**2, axis=1)),
    ("tessera.ebm.langevin_step", [(2, 3), (2, 3), (2, 3)], (2, 3), {"eta": 0.125, "noise_scale": 0.3}, lambda x: x[0] - .125*x[1] + .3*x[2]),
    ("tessera.ebm.refinement", [(2, 3), (2, 3)], (2, 3), {"eta": 0.125, "steps": 4}, lambda x: x[0] - .5*x[1]),
    ("tessera.ebm.partition_exact", [(2, 3)], (), {"temperature": 0.75, "reduction": "logsumexp"}, lambda x: np.sum(np.exp(-x[0] / .75))),
    ("tessera.clifford.geometric_product", [(2, 8), (2, 8)], (2, 8), {"p": 3, "q": 0}, lambda x: _clifford_reference(x)),
    ("tessera.cholesky", [(3, 3)], (3, 3), {}, lambda x: np.linalg.cholesky(x[0])),
    ("tessera.cholesky", [(2, 3, 3)], (2, 3, 3), {}, lambda x: np.linalg.cholesky(x[0])),
    ("tessera.cholesky_solve", [(3, 3), (3, 2)], (3, 2), {}, lambda x: np.linalg.solve(x[0], x[1])),
    ("tessera.cholesky_solve", [(2, 3, 3), (2, 3, 2)], (2, 3, 2), {}, lambda x: np.linalg.solve(x[0], x[1])),
    ("tessera.tri_solve", [(3, 3), (3, 2)], (3, 2), {"lower": True}, lambda x: np.linalg.solve(np.tril(x[0]), x[1])),
])
def test_descriptor_execute_compare_on_metal(op, shapes, out_shape, kwargs, reference):
    from tessera.compiler.driver import compile_graph_module
    from tessera.runtime import RuntimeArtifact, launch
    rng = np.random.default_rng(7)
    values = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    if op in {"tessera.cholesky", "tessera.cholesky_solve"}:
        values[0] = (values[0] @ np.swapaxes(values[0], -1, -2) + 3*np.eye(3, dtype=np.float32)).astype(np.float32)
    elif op == "tessera.tri_solve":
        values[0] = np.tril(values[0]) + 3*np.eye(3, dtype=np.float32)
    bundle = compile_graph_module(_value_module(op, shapes, out_shape, kwargs=kwargs), source_origin="apple-e2e-descriptor-test", target="apple_gpu", options={"package_native": True}, enable_tool_validation=False)
    art = RuntimeArtifact(metadata={"target":"apple_gpu", "compiler_path":"apple_native_descriptor"}, target_ir=bundle.target_ir.text, native_image=bundle.native_image, launch_descriptor=bundle.launch_descriptor)
    args = {f"a{i}": value for i, value in enumerate(values)}
    args["out"] = np.empty(out_shape, dtype=np.float32)
    result = launch(art, args)
    assert result["ok"], result
    np.testing.assert_allclose(result["output"], reference(values), rtol=2e-4, atol=2e-5)
    for _ in range(2):
        args["out"].fill(np.nan)
        replay = launch(art, args)
        assert replay["ok"], replay
        np.testing.assert_allclose(replay["output"], reference(values), rtol=2e-4, atol=2e-5)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize(("dtype", "rtol", "atol"), [
    ("fp32", 2e-4, 2e-5),
    ("fp16", 5e-2, 5e-2),
    pytest.param("bf16", 8e-2, 8e-2, marks=pytest.mark.skipif(
        __import__("importlib").util.find_spec("ml_dtypes") is None,
        reason="ml_dtypes is required for bf16 BMM coverage")),
])
def test_canonical_compile_default_consumes_bmm_descriptor_on_metal(dtype, rtol, atol):
    """Every BMM storage ABI replays through the canonical descriptor route."""
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.runtime import launch

    rng = np.random.default_rng(19)
    storage = (__import__("ml_dtypes").bfloat16 if dtype == "bf16"
               else {"fp32": np.float32, "fp16": np.float16}[dtype])
    a = rng.standard_normal((2, 4, 8)).astype(storage)
    b = rng.standard_normal((2, 8, 16)).astype(storage)
    result = canonical_compile(_module(dtype), target="apple_gpu")
    assert result.bundle.native_image is not None
    assert result.bundle.launch_descriptor is not None
    for _ in range(3):
        launched = launch(result.to_runtime_artifact(), {
            "a": a, "b": b, "out": np.empty((2, 4, 16), dtype=storage),
        })
        assert launched["ok"], launched
        np.testing.assert_allclose(
            launched["output"].astype(np.float32),
            a.astype(np.float32) @ b.astype(np.float32), rtol=rtol, atol=atol,
        )


@pytest.mark.hardware_apple_gpu
def test_canonical_compile_default_consumes_softmax_descriptor_on_metal():
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.runtime import launch

    x = np.random.default_rng(41).standard_normal((3, 7)).astype(np.float32)
    result = canonical_compile(_softmax_module(), target="apple_gpu")
    assert result.bundle.native_image is not None
    assert result.bundle.launch_descriptor is not None
    expected = np.exp(x - x.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    for _ in range(3):
        launched = launch(result.to_runtime_artifact(), {"x": x, "out": np.empty_like(x)})
        assert launched["ok"], launched
        np.testing.assert_allclose(launched["output"], expected, rtol=2e-4, atol=2e-5)


def _ppo_extended_reference(values, clip, kl_coef, entropy_coef):
    new, old, adv, mask, ref, entropy = values
    ratio = np.exp(new - old)
    loss = -np.minimum(ratio * adv, np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
    delta = ref - new
    loss = loss + kl_coef * (np.exp(delta) - delta - 1.0) - entropy_coef * entropy
    return np.sum(loss * mask) / max(float(np.sum(mask)), 1.0)


def _clifford_reference(values):
    from tessera.runtime import _clifford_geo_product_cl30_np

    return _clifford_geo_product_cl30_np(np, values[0], values[1])


@pytest.mark.parametrize("op,shapes,out_shape,kwargs,reference", [
    ("tessera.matmul", [(3, 4), (4, 5)], (3, 5), {}, lambda x: x[0] @ x[1]),
    ("tessera.gemm", [(3, 4), (4, 5)], (3, 5), {}, lambda x: x[0] @ x[1]),
    ("tessera.batched_gemm", [(2, 3, 4), (2, 4, 5)], (2, 3, 5), {}, lambda x: x[0] @ x[1]),
    ("tessera.cholesky", [(3, 3)], (3, 3), {}, lambda x: np.linalg.cholesky(x[0])),
    ("tessera.tri_solve", [(3, 3), (3, 2)], (3, 2), {"lower": True}, lambda x: np.linalg.solve(np.tril(x[0]), x[1])),
    ("tessera.cholesky_solve", [(3, 3), (3, 2)], (3, 2), {"lower": True}, lambda x: np.linalg.solve(x[0], x[1])),
])
def test_apple_cpu_descriptor_execute_compare(op, shapes, out_shape, kwargs, reference):
    # This test proves the Apple CPU *native* route.  The portable CI lane may
    # inspect descriptor construction above, but cannot establish native Apple
    # execution (and canonical_compile correctly rejects that claim off-host).
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    from tessera.compiler.driver import compile_graph_module
    from tessera.runtime import RuntimeArtifact, launch

    rng = np.random.default_rng(29)
    values = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    if op in {"tessera.cholesky", "tessera.cholesky_solve"}:
        values[0] = (values[0] @ values[0].T + 3 * np.eye(3, dtype=np.float32)).astype(np.float32)
    elif op == "tessera.tri_solve":
        values[0] = np.tril(values[0]) + 3 * np.eye(3, dtype=np.float32)
    bundle = compile_graph_module(
        _value_module(op, shapes, out_shape, kwargs=kwargs),
        source_origin="apple-cpu-e2e-descriptor-test", target="apple_cpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    artifact = RuntimeArtifact(
        metadata={"target": "apple_cpu", "compiler_path": "apple_cpu_native_descriptor"},
        target_ir=bundle.target_ir.text, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
    )
    args = {f"a{i}": value for i, value in enumerate(values)}
    args["out"] = np.empty(out_shape, dtype=np.float32)
    result = launch(artifact, args)
    assert result["ok"], result
    np.testing.assert_allclose(result["output"], reference(values), rtol=2e-4, atol=2e-5)
    args["out"].fill(np.nan)
    replay = launch(artifact, args)
    assert replay["ok"], replay
    np.testing.assert_allclose(replay["output"], reference(values), rtol=2e-4, atol=2e-5)


def test_apple_cpu_tuple_descriptor_states_are_explicit():
    from tessera.compiler.apple_cpu_native import value_descriptor_state

    assert value_descriptor_state("tessera.matmul") == "descriptor_ready"
    assert value_descriptor_state("tessera.gemm") == "descriptor_ready"
    assert value_descriptor_state("tessera.lu") == "descriptor_ready"
    assert value_descriptor_state("tessera.qr") == "descriptor_ready"
    assert value_descriptor_state("tessera.svd") == "descriptor_ready"


@pytest.mark.parametrize(("op", "shape", "results"), [
    ("tessera.lu", (4, 4), (("lu", (4, 4), "fp32"), ("piv", (4,), "int32"))),
    ("tessera.qr", (6, 4), (("q", (6, 4), "fp32"), ("r", (4, 4), "fp32"))),
    ("tessera.svd", (6, 4), (("u", (6, 4), "fp32"), ("s", (4,), "fp32"), ("vh", (4, 4), "fp32"))),
])
def test_apple_cpu_tuple_descriptor_execute_compare(op, shape, results):
    """Tuple outputs preserve SSA order and replay through one descriptor."""
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    from tessera.compiler.driver import compile_graph_module
    from tessera.runtime import RuntimeArtifact, launch

    rng = np.random.default_rng(53)
    a = rng.standard_normal(shape).astype(np.float32)
    bundle = compile_graph_module(
        _tuple_value_module(op, shape, results),
        source_origin="apple-cpu-e2e-tuple-descriptor-test", target="apple_cpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert [binding.name for binding in bundle.launch_descriptor.buffers if binding.direction == "output"] == [
        name for name, _, _ in results
    ]
    artifact = RuntimeArtifact(
        metadata={"target": "apple_cpu", "compiler_path": "apple_cpu_native_descriptor"},
        target_ir=bundle.target_ir.text, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
    )
    storage_dtype = {"fp32": np.float32, "int32": np.int32}
    args = {"a0": a}
    args.update({name: np.empty(shape, dtype=storage_dtype[dtype]) for name, shape, dtype in results})
    launched = launch(artifact, args)
    assert launched["ok"], launched
    output = launched["output"]
    assert isinstance(output, tuple) and len(output) == len(results)
    if op == "tessera.lu":
        lu, pivots = output
        lower = np.tril(lu, -1) + np.eye(shape[0], dtype=np.float32)
        upper = np.triu(lu)
        permuted = a.copy()
        for index, pivot in enumerate(pivots):
            permuted[[index, int(pivot) - 1]] = permuted[[int(pivot) - 1, index]]
        np.testing.assert_allclose(lower @ upper, permuted, rtol=1e-3, atol=1e-3)
    elif op == "tessera.qr":
        q, r = output
        np.testing.assert_allclose(q @ r, a, rtol=1e-3, atol=1e-3)
    else:
        u, s, vh = output
        np.testing.assert_allclose(u @ np.diag(s) @ vh, a, rtol=1e-3, atol=1e-3)
    for name, _, dtype in results:
        args[name].fill(0 if dtype == "int32" else np.nan)
    replay = launch(artifact, args)
    assert replay["ok"], replay


def test_canonical_compile_default_consumes_apple_cpu_bmm_descriptor():
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.runtime import launch

    rng = np.random.default_rng(31)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 4, 5)).astype(np.float32)
    result = canonical_compile(
        _value_module("tessera.batched_gemm", [(2, 3, 4), (2, 4, 5)], (2, 3, 5)),
        target="apple_cpu",
    )
    assert result.executable, result.reason
    assert result.bundle.launch_descriptor is not None
    launched = launch(result.to_runtime_artifact(), {"a0": a, "a1": b, "out": np.empty((2, 3, 5), np.float32)})
    assert launched["ok"], launched
    np.testing.assert_allclose(launched["output"], a @ b, rtol=2e-4, atol=2e-5)
