"""Stage L4 — the COMPILER-GENERATED ROCm GEMM through ``runtime.launch()``.

Stages L1–L3 made the compiler generate a correct, problem-generic, register-
blocked WMMA GEMM and serialize it to hsaco entirely in-process (no mlir-opt).
L4 wires that compiled path into the production runtime dispatch as an *opt-in*
lane: an artifact stamped ``compiler_path = "rocm_compiled"`` routes through the
execution matrix to ``_execute_rocm_compiled_gemm``, which generates + serializes
the kernel via tessera-opt and launches the hsaco on the AMD GPU.

This is the same ``runtime.launch()`` entry point the hand-written ``rocm_wmma``
lane uses — the only difference is *which kernel runs*: the one the Tessera
compiler emitted, not the hand-written HIPRTC kernel. The compiled lane is
opt-in; the hand-written lane stays the default + reference oracle (ROCM_AUDIT
L4). Here we prove the compiled lane executes through ``launch()`` and matches
both numpy AND the hand-written oracle (bit-identical).

Skip-clean: tessera-opt not built (or no in-process serialization), or no AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


# A file-defined jit fn so AST lowering works (decoration needs no GPU; the
# executability decision is taken later, in runtime_artifact(), and is
# host-gated).
@tessera.jit(target="rocm")
def _rocm_mm(a, b):
    return tessera.ops.matmul(a, b)


def test_execution_matrix_has_rocm_compiled_row():
    from tessera.compiler import execution_matrix as em
    row = em.lookup("rocm", "rocm_compiled")
    assert row is not None
    assert row.executor_id == "rocm_compiled"
    assert row.execution_kind == "native_gpu"


def test_rocm_compiled_executor_registered():
    from tessera import runtime as rt
    assert "rocm_compiled" in rt._executor_table()


def _compiled_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no AMD GPU / libtessera_rocm_gemm.so")
    return rt


def _artifact(rt, compiler_path):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": compiler_path, "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (256, 256, 256)])
def test_launch_rocm_compiled_matmul_f16_matches_numpy_and_oracle(shape):
    rt = _compiled_or_skip()
    m, n, k = shape
    rng = np.random.default_rng(5)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)

    # The compiled lane: tessera-opt generates + serializes the kernel in-process,
    # HIP launches the hsaco.
    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "rocm_compiled"
    assert res["execution_kind"] == "native_gpu"
    out = res["output"]

    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(out - ref))) < 5e-2, f"vs numpy {shape}"

    # vs the hand-written oracle through the SAME launch() entry point — the two
    # lanes compute the identical GEMM, so they agree bit-for-bit.
    oracle = rt.launch(_artifact(rt, "rocm_wmma"), (a, b))
    assert oracle["ok"] is True, oracle.get("reason")
    assert float(np.max(np.abs(out - oracle["output"]))) == 0.0, \
        f"compiled lane != hand-written oracle at {shape}"


def test_launch_rocm_compiled_bf16_matches_numpy_and_oracle():
    """bf16 storage (f32 accumulate): the generating pass emits bf16 fragments +
    the rocdl.wmma.*.bf16 intrinsic. Compare to numpy and the hand-written bf16
    oracle through the same launch() entry point."""
    rt = _compiled_or_skip()
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    m, n, k = 96, 80, 64
    rng = np.random.default_rng(9)
    a = (rng.standard_normal((m, k)) * 0.4).astype(bf16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(bf16)

    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_compiled"
    out = res["output"]

    ref = a.astype(np.float32) @ b.astype(np.float32)
    # bf16 has ~8 mantissa bits; tolerance scales with K.
    assert float(np.max(np.abs(out - ref))) < 5e-1, "bf16 vs numpy"

    oracle = rt.launch(_artifact(rt, "rocm_wmma"), (a, b))
    assert oracle["ok"] is True, oracle.get("reason")
    assert float(np.max(np.abs(out - oracle["output"]))) == 0.0, \
        "compiled bf16 lane != hand-written bf16 oracle"


@pytest.mark.parametrize("m,n,k", [(16, 16, 16), (64, 48, 32), (100, 96, 64),
                                   (40, 24, 48), (33, 17, 31)])
def test_launch_rocm_compiled_int8_matches_numpy(m, n, k):
    """int8 storage, i32 accumulate (rocdl iu8 WMMA, signed). Integer arithmetic
    is exact, so the compiled kernel must match numpy's int32 matmul EXACTLY —
    across aligned, ragged-M/N, and ragged-K shapes (the iu8 fragment is the 16
    int8 of a row/col bitcast to vector<4xi32>)."""
    rt = _compiled_or_skip()
    rng = np.random.default_rng(13 + m + n + k)
    a = rng.integers(-50, 51, size=(m, k)).astype(np.int8)
    b = rng.integers(-50, 51, size=(k, n)).astype(np.int8)
    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_compiled"
    out = res["output"]
    assert out.dtype == np.int32
    ref = a.astype(np.int32) @ b.astype(np.int32)
    assert np.array_equal(out, ref), f"int8 GEMM != numpy at {m}x{n}x{k}"


def test_launch_rocm_compiled_rejects_f32():
    """f32 in is not a WMMA storage dtype — structured invalid_artifact, never a
    silent miscompute."""
    rt = _compiled_or_skip()
    a = np.zeros((32, 32), np.float32)
    b = np.zeros((32, 32), np.float32)
    res = rt.launch(_artifact(rt, "rocm_compiled"), (a, b))
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "f16/bf16" in res["reason"]


# ── Stage L4 default flip: @jit(target="rocm") matmul -> compiled lane ────────

def test_jit_rocm_matmul_defaults_to_compiled_lane():
    """On a capable host the rocm matmul artifact is executable through the
    COMPILER-GENERATED lane by default (was artifact_only). Off-device it stays
    artifact_only — so this asserts the flip only where the lane can run."""
    rt = _compiled_or_skip()
    art = _rocm_mm.runtime_artifact()
    md = art.metadata or {}
    assert md.get("executable") is True
    assert md.get("compiler_path") == "rocm_compiled"
    assert md.get("execution_kind") == "native_gpu"
    assert md.get("rocm_fallback_lane") == "rocm_wmma"

    m, n, k = 128, 96, 64
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
    res = rt.launch(art, (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_compiled"
    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(res["output"] - ref))) < 5e-2


def test_compiled_lane_falls_back_to_oracle_when_unavailable(monkeypatch):
    """When the compiled lane can't run (here: tessera-opt forced absent), the
    executor degrades to the hand-written rocm_wmma oracle — so flipping the
    default never regresses availability. A genuine kernel failure is NOT masked
    (only _RocmCompiledUnavailable triggers the fallback)."""
    rt = _compiled_or_skip()
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: None)
    rt._rocm_compiled_hsaco_cache.clear()  # don't serve a cached hsaco
    m, n, k = 64, 48, 32
    rng = np.random.default_rng(4)
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
    # Direct executor call: compiled build raises _RocmCompiledUnavailable ->
    # falls back to the hand-written oracle, which returns the correct result.
    out = rt._execute_rocm_compiled_gemm(_artifact(rt, "rocm_compiled"), (a, b))
    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(out - ref))) < 1e-2
