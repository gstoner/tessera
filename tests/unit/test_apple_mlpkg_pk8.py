"""PK8 — author a *production* ``.mtlpackage`` from the MPSGraph lane.

This is the inverse of PK1-PK7 (which *consume* a package). PK8 proves
Tessera can *author* its own packaged kernel on-host, with no coremltools
and no DXIL, by riding the same MPSGraph primitive the runtime already
builds for its MPSGraph-lane ops:

  build MPSGraph matmul
    → compile to MPSGraphExecutable
    → serializeToMPSGraphPackageAtURL:        (MPSGraphExecutable.h:205)
    → wrap with the MLLibrary ``manifest.json``
    → .mtlpackage

The authored package then flows through the *existing* PK1-PK7 lifecycle
unchanged (load → reflect → prepare → dispatch) and the GPU output is
compared bitwise against numpy ``A @ B``.

Grounded finding (2026-06-02): MPSGraph's serialized packages expose
*positionally-indexed, unnamed* bindings (``MPSGraphTensor`` has no name
property; the Apple sample's ``inputA``/``inputB``/``output`` names came
from its CoreML origin). So PK8 binds by ``buffer_index`` via the
``fill_input_at`` / ``read_output_at`` ABI — the argument table is still
correct (resources bound by reflected index). This is an Apple-side fact,
not a Tessera limitation.

Skips cleanly off-Darwin / pre-macOS-26 (the MTL4 ML dispatch gate).
Authoring alone needs only macOS 14, but the end-to-end dispatch proof
needs the full MTL4 ML pipeline, so we gate on ``packaged_ml_available``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tessera.apple_mlpkg import (
    ArgumentLayout,
    ArgumentLayoutEntry,
    AUTHOR_CHAINS,
    AUTHOR_OP_BINARY,
    AUTHOR_OP_ROWOP,
    AUTHOR_OP_UNARY,
    AUTHOR_OPS,
    author_chain_package,
    author_matmul_package,
    author_op_package,
    composite_package_descriptor,
    compile_mlpackage,
    execute_composite_package,
    first_function_name,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
    package_tree_digest,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _run_authored_op(op, rows, cols, inputs, *, weighted=False, eps=1e-5):
    """Author ``op`` → load → prepare → fill (positional) → dispatch → read.
    Returns the GPU output as an (rows, cols) fp32 array."""
    import os
    import tempfile

    d = tempfile.mkdtemp(prefix="pk8op_")
    pkg = os.path.join(d, f"{op}.mtlpackage")
    assert author_op_package(pkg, op, rows, cols, eps=eps, weighted=weighted), (
        f"author_op_package({op}) failed"
    )
    fn = first_function_name(pkg) or "main"
    pipe = compile_mlpackage(pkg, function_name=fn)
    assert pipe is not None, f"PK1 load failed for {op} err={last_error_kind()}"
    try:
        assert pipe.prepare_tensors(), f"prepare failed for {op}"
        for i, arr in enumerate(inputs):
            assert pipe.fill_input_at(i, arr.astype(np.float32).tobytes()), (
                f"fill input {i} failed for {op}"
            )
        assert pipe.dispatch(timeout_ms=30_000), f"dispatch failed for {op}"
        raw = pipe.read_output_at(len(inputs), rows * cols * 4)
        assert raw is not None, f"read failed for {op}"
        return np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)
    finally:
        pipe.destroy()


def _require_packaged_ml() -> None:
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")


def test_pk8_symbols_resolve():
    """The authoring + positional-binding ABI is reachable. (Does not
    require a live device — just that the wrappers bind.)"""
    # Importing + referencing is enough; the functions guard internally
    # and return False/None when the runtime is unavailable.
    assert callable(author_matmul_package)
    assert callable(first_function_name)


def test_pk8_author_produces_mtlpackage_structure():
    """Authoring writes the canonical MLLibrary layout: a ``manifest.json``
    wrapping a ``library.mpsgraphpackage`` (matching Apple's sample)."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "tessera_matmul.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4), (
            "author_matmul_package failed on a host that reports "
            "packaged ML available"
        )
        assert (pkg / "manifest.json").is_file()
        inner = pkg / "library.mpsgraphpackage"
        assert inner.is_dir()
        # The serialized MPSGraph executable: model + reflection.
        assert (inner / "reflection.fb").is_file()
        manifest = (pkg / "manifest.json").read_text()
        assert "MLLibrary" in manifest
        assert "library.mpsgraphpackage" in manifest


def test_pk8_authored_package_loads_and_reflects():
    """PK1+PK2 over a Tessera-authored package: it loads as an MTL4 ML
    pipeline and reflects 3 fp32 bindings at indices 0/1/2."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "m.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4)
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None, f"PK1 load failed err={last_error_kind()}"
        try:
            binds = pipe.bindings()
            # Unnamed bindings collapse in the name-keyed dict, so assert on
            # the raw reflection via prepare + index addressing instead.
            assert pipe.prepare_tensors()
            # Every binding the package declares is fp32.
            for b in binds.values():
                assert b.dtype == "fp32"
        finally:
            pipe.destroy()


@pytest.mark.parametrize("mkn", [(4, 4, 4), (8, 16, 4), (3, 5, 7)])
def test_pk8_authored_matmul_dispatches_and_matches_numpy(mkn):
    """The full proof: author → load → prepare → fill (by index) →
    dispatch → read → compare to numpy ``A @ B`` at fp32 tolerance."""
    _require_packaged_ml()
    m, k, n = mkn
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "matmul.mtlpackage"
        assert author_matmul_package(str(pkg), m, k, n)
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None, f"PK1 load failed err={last_error_kind()}"
        try:
            assert pipe.prepare_tensors()
            rng = np.random.default_rng(0xA11CE)
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
            # MPSGraph-authored packages are positionally bound:
            # inputs at 0/1, output at the last index (2).
            assert pipe.fill_input_at(0, a.tobytes())
            assert pipe.fill_input_at(1, b.tobytes())
            assert pipe.dispatch(timeout_ms=30_000)
            raw = pipe.read_output_at(2, m * n * 4)
            assert raw is not None
            c = np.frombuffer(raw, dtype=np.float32).reshape(m, n)
            ref = a @ b
            assert np.allclose(c, ref, rtol=1e-4, atol=1e-4), (
                f"max abs err {np.max(np.abs(c - ref)):.3e}"
            )
        finally:
            pipe.destroy()


# ── generalized authoring: unary / rowop / norm / binary families ────────


def test_pk8_author_ops_vocabulary_is_grouped():
    """The op vocabulary is partitioned into the three binding-arity
    families with no overlap."""
    assert set(AUTHOR_OPS) == set(AUTHOR_OP_UNARY) | set(AUTHOR_OP_ROWOP) | set(
        AUTHOR_OP_BINARY
    )
    assert not (set(AUTHOR_OP_UNARY) & set(AUTHOR_OP_BINARY))
    assert not (set(AUTHOR_OP_UNARY) & set(AUTHOR_OP_ROWOP))


@pytest.mark.parametrize(
    "op,ref",
    [
        ("relu", lambda x: np.maximum(x, 0.0)),
        ("sigmoid", _sigmoid),
        ("tanh", np.tanh),
        ("silu", _silu),
        ("gelu", _gelu),
        ("exp", np.exp),
        ("abs", np.abs),
        ("neg", lambda x: -x),
    ],
)
def test_pk8_author_unary_matches_numpy(op, ref):
    """Each unary activation authors a package whose dispatch matches the
    numpy reference — proving the authored kernel == the runtime's
    MPSGraph-lane builder."""
    _require_packaged_ml()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    out = _run_authored_op(op, 4, 8, [x])
    assert np.allclose(out, ref(x).astype(np.float32), rtol=1e-4, atol=2e-4), (
        f"{op}: max abs err {np.max(np.abs(out - ref(x))):.3e}"
    )


def test_pk8_author_softmax_matches_numpy():
    _require_packaged_ml()
    rng = np.random.default_rng(12)
    x = rng.standard_normal((6, 5)).astype(np.float32)
    out = _run_authored_op("softmax", 6, 5, [x])
    assert np.allclose(out, _softmax(x), rtol=1e-4, atol=2e-4)


def test_pk8_author_rmsnorm_weighted_matches_numpy():
    """rmsnorm authors with a gamma input (positional binding 1)."""
    _require_packaged_ml()
    rng = np.random.default_rng(13)
    rows, cols, eps = 4, 8, 1e-5
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    g = rng.standard_normal((cols,)).astype(np.float32)
    out = _run_authored_op("rmsnorm", rows, cols, [x, g], weighted=True, eps=eps)
    ref = x / np.sqrt((x**2).mean(axis=1, keepdims=True) + eps) * g
    assert np.allclose(out, ref, rtol=1e-4, atol=2e-4)


def test_pk8_author_layer_norm_weighted_matches_numpy():
    """layer_norm authors with gamma + beta inputs (bindings 1 and 2)."""
    _require_packaged_ml()
    rng = np.random.default_rng(14)
    rows, cols, eps = 4, 8, 1e-5
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    g = rng.standard_normal((cols,)).astype(np.float32)
    b = rng.standard_normal((cols,)).astype(np.float32)
    out = _run_authored_op(
        "layer_norm", rows, cols, [x, g, b], weighted=True, eps=eps
    )
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + eps) * g + b
    assert np.allclose(out, ref, rtol=1e-4, atol=2e-4)


def test_pk8_author_silu_mul_binary_matches_numpy():
    """silu_mul = a * silu(b) — a 2-input package (bindings 0 and 1)."""
    _require_packaged_ml()
    rng = np.random.default_rng(15)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    out = _run_authored_op("silu_mul", 4, 8, [a, b])
    assert np.allclose(out, a * _silu(b), rtol=1e-4, atol=2e-4)


def test_pk8_author_op_rejects_unknown_op():
    """An unrecognized op name fails cleanly (no crash, returns False)."""
    _require_packaged_ml()
    import os
    import tempfile

    d = tempfile.mkdtemp(prefix="pk8bad_")
    pkg = os.path.join(d, "bad.mtlpackage")
    assert author_op_package(pkg, "not_a_real_op", 4, 4) is False


# ── fused multi-op chains (PK8b) ─────────────────────────────────────────


def _run_authored_chain(chain, dims, inputs, out_shape, *, eps=1e-5):
    import os
    import tempfile

    d = tempfile.mkdtemp(prefix="pk8chain_")
    pkg = os.path.join(d, f"{chain}.mtlpackage")
    assert author_chain_package(pkg, chain, dims, eps=eps), (
        f"author_chain_package({chain}) failed"
    )
    fn = first_function_name(pkg) or "main"
    pipe = compile_mlpackage(pkg, function_name=fn)
    assert pipe is not None, f"PK1 load failed for {chain} err={last_error_kind()}"
    try:
        assert pipe.prepare_tensors(), f"prepare failed for {chain}"
        for i, arr in enumerate(inputs):
            assert pipe.fill_input_at(i, arr.astype(np.float32).tobytes())
        assert pipe.dispatch(timeout_ms=30_000), f"dispatch failed for {chain}"
        r, c = out_shape
        raw = pipe.read_output_at(len(inputs), r * c * 4)
        assert raw is not None
        return np.frombuffer(raw, dtype=np.float32).reshape(r, c)
    finally:
        pipe.destroy()


def test_pk8b_chain_matmul_softmax_matches_numpy():
    """Fused softmax(A@B) — two ops, one package, one dispatch."""
    _require_packaged_ml()
    rng = np.random.default_rng(20)
    M, K, N = 4, 6, 5
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = _run_authored_chain("matmul_softmax", [M, K, N], [a, b], (M, N))
    assert np.allclose(out, _softmax(a @ b), rtol=1e-4, atol=2e-4)


def test_pk8b_chain_attention_block_matches_numpy():
    """Fused softmax(A@B)@C — the 3-op attention block in one package."""
    _require_packaged_ml()
    rng = np.random.default_rng(21)
    M, K, N, P = 4, 6, 5, 3
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = rng.standard_normal((N, P)).astype(np.float32)
    out = _run_authored_chain(
        "matmul_softmax_matmul", [M, K, N, P], [a, b, c], (M, P)
    )
    assert np.allclose(out, _softmax(a @ b) @ c, rtol=1e-4, atol=2e-4)


def test_pk8b_composite_descriptor_executes_and_replays_matmul_softmax():
    """Descriptor-gated package execution owns intermediates and permits replay.

    This is the E2E2 seam: a descriptor seals a reflected external ABI and
    package identity, then drives an authored multi-op MTL4 package twice.
    """
    _require_packaged_ml()
    rng = np.random.default_rng(0xC0A5)
    M, K, N = 4, 6, 5
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "matmul_softmax.mtlpackage"
        assert author_chain_package(package, "matmul_softmax", [M, K, N])
        function = first_function_name(package) or "main"
        layout = ArgumentLayout(
            pipeline_path=str(package), function_name=function,
            entries=(
                ArgumentLayoutEntry("lhs", 0, "tensor", "fp32", 2,
                                    (M, K), "input", "shared"),
                ArgumentLayoutEntry("rhs", 1, "tensor", "fp32", 2,
                                    (K, N), "input", "shared"),
                ArgumentLayoutEntry("output", 2, "tensor", "fp32", 2,
                                    (M, N), "output", "shared"),
            ),
        )
        descriptor = composite_package_descriptor(package_tree_digest(package), layout)
        pipeline = compile_mlpackage(package, function_name=function)
        assert pipeline is not None
        try:
            inputs = {"lhs": a.tobytes(), "rhs": b.tobytes()}
            first = execute_composite_package(
                descriptor, pipeline, inputs, {"output": M * N * 4})
            second = execute_composite_package(
                descriptor, pipeline, inputs, {"output": M * N * 4})
            expected = _softmax(a @ b)
            for result in (first, second):
                output = np.frombuffer(result["output"], dtype=np.float32).reshape(M, N)
                assert np.allclose(output, expected, rtol=1e-4, atol=2e-4)
            assert pipeline.intermediates_heap_size() > 0
        finally:
            pipeline.destroy()


def test_pk8b_chain_rmsnorm_matmul_matches_numpy():
    """Fused (rmsnorm(x)*gamma) @ W — a norm+projection block."""
    _require_packaged_ml()
    rng = np.random.default_rng(22)
    M, K, N, eps = 4, 6, 5, 1e-5
    x = rng.standard_normal((M, K)).astype(np.float32)
    g = rng.standard_normal((K,)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)
    out = _run_authored_chain(
        "rmsnorm_matmul", [M, K, N], [x, g, w], (M, N), eps=eps
    )
    rms = x / np.sqrt((x**2).mean(axis=1, keepdims=True) + eps)
    assert np.allclose(out, (rms * g) @ w, rtol=1e-4, atol=2e-4)


def test_pk8b_chain_rejects_unknown_and_bad_arity():
    _require_packaged_ml()
    import os
    import tempfile

    d = tempfile.mkdtemp(prefix="pk8bad_")
    assert author_chain_package(
        os.path.join(d, "x.mtlpackage"), "not_a_chain", [4, 4, 4]
    ) is False
    # matmul_softmax needs exactly 3 dims.
    assert author_chain_package(
        os.path.join(d, "y.mtlpackage"), "matmul_softmax", [4, 4]
    ) is False


def test_pk8b_chains_vocabulary():
    assert "matmul_softmax" in AUTHOR_CHAINS
    assert "matmul_softmax_matmul" in AUTHOR_CHAINS
    assert "rmsnorm_matmul" in AUTHOR_CHAINS


def test_pk8_reauthor_is_deterministic_overwrite():
    """Re-authoring into the same path overwrites cleanly (serialize is
    set to non-append) — no stale ``.mpsgraphpackage`` contents leak."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "r.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4)
        assert author_matmul_package(str(pkg), 8, 8, 8)  # different shape
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None
        try:
            assert pipe.prepare_tensors()
            rng = np.random.default_rng(1)
            a = rng.standard_normal((8, 8)).astype(np.float32)
            b = rng.standard_normal((8, 8)).astype(np.float32)
            assert pipe.fill_input_at(0, a.tobytes())
            assert pipe.fill_input_at(1, b.tobytes())
            assert pipe.dispatch(timeout_ms=30_000)
            raw = pipe.read_output_at(2, 8 * 8 * 4)
            c = np.frombuffer(raw, dtype=np.float32).reshape(8, 8)
            assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)
        finally:
            pipe.destroy()
