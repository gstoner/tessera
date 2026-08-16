"""The Apple dylib identity guard must stay strong *and* stop dominating launches.

The check is load-bearing: it is what stops a stale or substituted dylib from
masquerading as the image a descriptor was built against — a mistake this backend
makes easily, because rebuilding the runtime republishes it under a new cache
path. It previously re-read and SHA-256'd the whole ~1 MB dylib on every launch,
which measured ~580 us against ~23 us of actual GPU work for a row softmax.

These tests pin both halves: the guard still discriminates, and it is memoized on
file identity so a launch pays a `stat` rather than a full re-hash.
"""
from __future__ import annotations

import hashlib
import os

import pytest

import tessera.runtime as rt


def _apple_dylib():
    from tessera.compiler.apple_native import _runtime_library_path

    return _runtime_library_path()


_DYLIB = _apple_dylib()
_needs_dylib = pytest.mark.skipif(
    _DYLIB is None, reason="Apple GPU runtime dylib unavailable"
)


@_needs_dylib
def test_identity_guard_accepts_the_real_digest_and_rejects_others() -> None:
    """Memoizing must not turn the guard into a rubber stamp."""
    real = hashlib.sha256(_DYLIB.read_bytes()).hexdigest()
    assert rt._apple_runtime_identity_matches(real) is True
    assert rt._apple_runtime_identity_matches("0" * 64) is False
    # A wrong digest must stay wrong on the cached path too, not just the first
    # (uncached) call — the cache keys the *file*, never the expectation.
    assert rt._apple_runtime_identity_matches("0" * 64) is False
    assert rt._apple_runtime_identity_matches(real) is True


@_needs_dylib
def test_identity_guard_rehashes_when_the_file_identity_changes() -> None:
    """A rebuild or republish must re-verify content, not serve a stale answer."""
    real = hashlib.sha256(_DYLIB.read_bytes()).hexdigest()
    rt._APPLE_RUNTIME_DIGEST_CACHE.clear()
    assert rt._apple_runtime_identity_matches(real) is True
    first = dict(rt._APPLE_RUNTIME_DIGEST_CACHE)
    assert len(first) == 1
    os.utime(_DYLIB, None)  # what a rebuild does to mtime
    assert rt._apple_runtime_identity_matches(real) is True
    # A new key means the changed file was re-hashed rather than assumed good.
    assert set(rt._APPLE_RUNTIME_DIGEST_CACHE) != set(first)


@_needs_dylib
def test_identity_guard_cache_stays_bounded() -> None:
    """Entries are only added on genuine identity changes; still, bound it."""
    rt._APPLE_RUNTIME_DIGEST_CACHE.clear()
    real = hashlib.sha256(_DYLIB.read_bytes()).hexdigest()
    for _ in range(40):
        os.utime(_DYLIB, None)
        rt._apple_runtime_identity_matches(real)
    assert len(rt._APPLE_RUNTIME_DIGEST_CACHE) <= 17


@pytest.mark.hardware_apple_gpu
@_needs_dylib
def test_launch_still_produces_a_correct_native_result() -> None:
    """The point that matters: a faster launch must still be a *working* launch.

    Regression for a near-miss during this work — an early version of the cache
    raised a NameError that `launch` swallowed into `ok: False`, and the "62x
    speedup" it appeared to produce was the cost of failing fast with an
    all-zero output. Speed assertions are worthless without a correctness
    assertion beside them.
    """
    import numpy as np

    from tessera.compiler import apple_native
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    from tessera.compiler.scheduled_matmul import find_tessera_opt

    if not apple_native.tools_available() or find_tessera_opt() is None:
        pytest.skip("Apple GPU runtime dylib / tessera-opt unavailable")
    x_type = IRType("tensor<64x256xf32>", ("64", "256"), "fp32")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="identity_cache_softmax", args=[IRArg("x", x_type)],
        result_types=[x_type],
        body=[IROp(result="o", op_name="tessera.softmax", operands=["%x"],
                   operand_types=[str(x_type)], result_type=str(x_type),
                   kwargs={"axis": -1})],
        return_values=["%o"])])
    bundle = compile_graph_module(
        module, source_origin="identity-cache", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text if bundle.tile else "",
        target_ir=bundle.target_ir.text if bundle.target_ir else "",
    )
    rng = np.random.default_rng(31)
    x = np.ascontiguousarray(rng.standard_normal((64, 256)), dtype=np.float32)
    out = np.zeros_like(x)
    # Repeat so the cached path — not just the first, uncached one — is proven.
    for _ in range(3):
        out[:] = 0.0
        result = rt.launch(artifact, {"x": x, "o": out})
        assert result["ok"] is True, result.get("reason")
        assert result.get("execution_kind") == "native_gpu", result
        shifted = x - x.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(out, expected, rtol=3e-5, atol=3e-5)
