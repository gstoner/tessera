from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


def test_jit_matmul_executes_through_cpu_lowering_path():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(12, dtype=np.float32).reshape(3, 4)

    out = mm(A, B)

    np.testing.assert_allclose(out, A @ B)
    assert mm.cpu_plan is not None
    assert mm.cpu_plan.op_name == "tessera.matmul"
    assert mm.is_executable
    assert "JIT_COMPILED_CPU" in mm.explain_lowering()


def test_jit_gemm_lowering_artifacts_cover_all_compiler_layers():
    @ts.jit
    def gemm(A, B):
        return ts.ops.gemm(A, B)

    artifacts = {artifact.level: artifact.text for artifact in gemm.lowering_artifacts()}

    assert set(artifacts) == {"graph", "schedule", "tile", "target"}
    assert "tessera.matmul" in artifacts["graph"]
    assert "tessera.gemm" not in artifacts["graph"]
    assert "schedule.tile" in artifacts["schedule"]
    assert "tile_m = 128" in artifacts["schedule"]
    assert "tile.mma" in artifacts["tile"]
    assert "tessera.cpu.matmul" in artifacts["target"]
    assert gemm.schedule_ir == artifacts["schedule"]
    assert gemm.tile_ir == artifacts["tile"]
    assert gemm.target_ir == artifacts["target"]
    assert gemm.runtime_artifact().metadata["compiler_path"] == "jit_cpu_numpy"


def test_jit_gemm_can_use_256x128_cpu_tile():
    @ts.jit(cpu_tile=(256, 128, 64))
    def gemm_256x128(A, B):
        return ts.ops.gemm(A, B)

    A = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    B = np.arange(64 * 128, dtype=np.float32).reshape(64, 128)

    np.testing.assert_allclose(gemm_256x128(A, B), A @ B, rtol=1e-4)
    assert gemm_256x128.is_executable
    assert gemm_256x128.cpu_plan.tile == (256, 128, 64)
    assert "tile_m = 256" in gemm_256x128.schedule_ir
    assert "tile_n = 128" in gemm_256x128.schedule_ir
    assert "tile_k = 64" in gemm_256x128.schedule_ir


def test_jit_dynamic_function_without_source_reports_uninspectable_fallback():
    ns = {}
    exec(
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.gemm(A, B)\n",
        {"ts": ts},
        ns,
    )

    dynamic_mm = ts.jit(ns["dynamic_mm"])

    assert not dynamic_mm.is_executable
    explanation = dynamic_mm.explain_lowering()
    assert "JIT_SOURCE_UNAVAILABLE" in explanation
    assert "JIT_EAGER_FALLBACK" in explanation


def test_jit_dynamic_function_can_compile_with_explicit_source():
    source = (
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.gemm(A, B)\n"
    )
    ns = {}
    exec(source, {"ts": ts}, ns)

    dynamic_mm = ts.jit(ns["dynamic_mm"], source=source, cpu_tile=(256, 128, 64))
    A = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    B = np.arange(64 * 128, dtype=np.float32).reshape(64, 128)

    np.testing.assert_allclose(dynamic_mm(A, B), A @ B, rtol=1e-4)
    assert dynamic_mm.is_executable
    assert dynamic_mm.source_origin == "explicit"
    assert "JIT_SOURCE_PROVIDED" in dynamic_mm.explain_lowering()
    assert "JIT_COMPILED_CPU" in dynamic_mm.explain_lowering()
    assert "tile_m = 256" in dynamic_mm.schedule_ir


def test_jit_dynamic_function_effects_use_explicit_source_for_determinism():
    source = (
        "def dynamic_dropout(x):\n"
        "    return ts.ops.dropout(x)\n"
    )
    ns = {}
    exec(source, {"ts": ts}, ns)

    with pytest.raises(ts.TesseraEffectError):
        ts.jit(ns["dynamic_dropout"], source=source, deterministic=True)


def test_jit_matmul_cpu_plan_supports_kwargs():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)

    np.testing.assert_allclose(mm(B=B, A=A), A @ B)


def test_unary_ops_compile_through_cpu_path():
    @ts.jit
    def relu(x):
        return ts.ops.relu(x)

    @ts.jit
    def sigmoid(x):
        return ts.ops.sigmoid(x)

    @ts.jit
    def sin(x):
        return ts.ops.sin(x)

    x = np.array([-1.0, 2.0], dtype=np.float32)

    np.testing.assert_allclose(relu(x), np.array([0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(sigmoid(x), 1.0 / (1.0 + np.exp(-x)))
    np.testing.assert_allclose(sin(x), np.sin(x))
    assert relu.is_executable
    assert sigmoid.is_executable
    assert sin.is_executable
    assert "tessera.cpu.relu" in relu.target_ir


def test_softmax_compiles_as_stable_cpu_reduction():
    @ts.jit
    def probs(x):
        return ts.ops.softmax(x)

    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]], dtype=np.float32)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))

    np.testing.assert_allclose(probs(x), e / np.sum(e, axis=-1, keepdims=True))
    assert probs.is_executable
    assert "stable_reduction" in probs.tile_ir


def test_adam_compiles_as_functional_cpu_optimizer_step():
    @ts.jit
    def adam_step(param, grad, m, v):
        return ts.ops.adam(param, grad, m, v)

    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    m = np.zeros_like(param)
    v = np.zeros_like(param)

    new_param, new_m, new_v = adam_step(param, grad, m, v)

    expected_m = 0.1 * grad
    expected_v = 0.001 * grad * grad
    expected_param = param - 1e-3 * grad / (np.sqrt(grad * grad) + 1e-8)
    np.testing.assert_allclose(new_m, expected_m, rtol=1e-6)
    np.testing.assert_allclose(new_v, expected_v, rtol=1e-6)
    np.testing.assert_allclose(new_param, expected_param, rtol=1e-6)
    assert adam_step.is_executable
    assert "functional_optimizer_step" in adam_step.tile_ir


def test_composite_supported_ops_compile_through_cpu_dataflow():
    @ts.jit
    def composite(x):
        y = ts.ops.relu(x)
        return ts.ops.softmax(y)

    x = np.array([-1.0, 2.0], dtype=np.float32)
    out = composite(x)

    assert composite.is_executable
    assert "JIT_COMPILED_CPU" in composite.explain_lowering()
    assert "tessera.cpu.relu" in composite.target_ir
    assert "tessera.cpu.softmax" in composite.target_ir
    np.testing.assert_allclose(out, ts.ops.softmax(ts.ops.relu(x)))


def test_nested_ops_and_keyword_literals_compile_through_cpu_dataflow():
    @ts.jit
    def nested(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B), axis=0)

    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32)
    scores = A @ B
    e = np.exp(scores - np.max(scores, axis=0, keepdims=True))

    np.testing.assert_allclose(nested(A, B), e / np.sum(e, axis=0, keepdims=True))
    assert nested.is_executable
    assert "axis = 0" in nested.ir_text()
    assert "tessera.cpu.matmul" in nested.target_ir
    assert "tessera.cpu.softmax" in nested.target_ir


def test_developer_frontend_docs_link_first_end_to_end_path():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    doc = root / "docs" / "guides" / "Tessera_Developer_Frontend_End_To_End.md"
    readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    api = (root / "docs" / "spec" / "PYTHON_API_SPEC.md").read_text(encoding="utf-8")

    text = doc.read_text(encoding="utf-8")
    assert "@jit supported op graph -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution" in text
    assert "mm.lowering_artifacts()" in text
    assert "eager Python fallback" in text
    assert "docs/guides/Tessera_Developer_Frontend_End_To_End.md" in readme
    assert ".lowering_artifacts()" in api


# ─────────────────────────────────────────────────────────────────────────────
# Item #1 (P0): hard-fail when @jit(target=...) is requested but source is
# uninspectable. Without source no Graph IR is emitted and __call__ would
# silently fall back to eager Python — looking like the target is active while
# actually returning numpy semantics.
# ─────────────────────────────────────────────────────────────────────────────


def test_jit_target_apple_cpu_without_source_raises_jit_error():
    ns = {}
    exec(
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.matmul(A, B)\n",
        {"ts": ts},
        ns,
    )

    with pytest.raises(ts.TesseraJitError) as excinfo:
        ts.jit(ns["dynamic_mm"], target="apple_cpu")

    msg = str(excinfo.value)
    assert "apple_cpu" in msg
    assert "dynamic_mm" in msg
    # The error must point the developer at the escape hatch.
    assert "source=" in msg or "source_path=" in msg
    assert "silently fall back" in msg.lower() or "silently" in msg.lower()


def test_jit_target_apple_gpu_without_source_raises_jit_error():
    ns = {}
    exec(
        "def dynamic_attn(q, k, v):\n"
        "    return ts.ops.flash_attn(q, k, v)\n",
        {"ts": ts},
        ns,
    )

    with pytest.raises(ts.TesseraJitError, match="apple_gpu"):
        ts.jit(ns["dynamic_attn"], target="apple_gpu")


def test_jit_target_rocm_without_source_raises_jit_error():
    ns = {}
    exec(
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.matmul(A, B)\n",
        {"ts": ts},
        ns,
    )

    with pytest.raises(ts.TesseraJitError, match="rocm"):
        ts.jit(ns["dynamic_mm"], target="rocm")


def test_jit_target_apple_cpu_with_explicit_source_compiles():
    # Counterpart to the failure above — providing source= unblocks the
    # native target path.
    source = (
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.matmul(A, B)\n"
    )
    ns = {}
    exec(source, {"ts": ts}, ns)

    mm = ts.jit(ns["dynamic_mm"], target="apple_cpu", source=source)

    assert mm.is_executable is True
    assert mm.runtime_artifact().metadata["compiler_path"] == "apple_cpu_accelerate"

    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 4), dtype=np.float32)
    np.testing.assert_allclose(mm(A, B), A @ B)


def test_jit_target_apple_cpu_with_source_path_compiles(tmp_path):
    src = tmp_path / "_dyn_mm.py"
    src.write_text(
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.matmul(A, B)\n",
        encoding="utf-8",
    )
    ns = {}
    exec(src.read_text(encoding="utf-8"), {"ts": ts}, ns)

    mm = ts.jit(ns["dynamic_mm"], target="apple_cpu", source_path=str(src))

    assert mm.is_executable is True
    assert mm.runtime_artifact().metadata["compiler_path"] == "apple_cpu_accelerate"


def test_jit_target_none_without_source_still_falls_back_softly():
    # target=None retains the existing soft-warning + eager Python behavior so
    # REPL/heredoc exploration of the default path stays ergonomic.
    ns = {}
    exec(
        "def dynamic_mm(A, B):\n"
        "    return ts.ops.matmul(A, B)\n",
        {"ts": ts},
        ns,
    )
    mm = ts.jit(ns["dynamic_mm"])  # no target

    assert not mm.is_executable
    assert "JIT_SOURCE_UNAVAILABLE" in mm.explain_lowering()
    A = np.eye(2, dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    np.testing.assert_allclose(mm(A, B), A @ B)  # eager numpy
