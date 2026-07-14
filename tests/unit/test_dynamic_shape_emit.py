"""Workstream G / W2 — the dynamic-shape emitter for the generic lanes.

The generic scalar kernels (x86 C, ROCm HIP, NVIDIA CUDA) already take M/N/K as
runtime args with in-kernel bounds guards, so their *source* is dims-invariant —
one compiled kernel serves every shape. `SpecPolicy.DYNAMIC` was defined from day
one (the interface half of W2) but the emitters raised on it; enabling it is the
implementation half:

* BUCKET keys the arbiter/AOT cache per shape-bucket → a separate (identical!)
  compile per bucket.
* DYNAMIC keys by the *symbolic* identity → ONE cache entry serves all shapes, so
  variable-shape serving (seq-len / batch / KV-len) never recompiles.

The bucket-specialized tensor-core lanes (WMMA / mma.sync) legitimately stay
BUCKET; DYNAMIC is for the generic middle-ground lanes W2 targets.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.rocm_hip as rocm  # noqa: F401 — self-registers
import tessera.compiler.emit.x86_llvm  # noqa: F401
import tessera.compiler.emit.nvidia_cuda  # noqa: F401
from tessera.compiler.emit import kernel_cache as KC
from tessera.compiler.emit.kernel_emitter import EmitError, SpecPolicy, get_emitter

_GENERIC_TARGETS = ("rocm", "x86", "nvidia")


def _rocm_hip_live() -> bool:
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        return False
    try:
        from tessera import runtime as rt
        return rt._rocm_wmma_runtime_available()
    except Exception:
        return False


# ── 1. DYNAMIC emit is accepted and dims-invariant (host-free) ────────────────

@pytest.mark.parametrize("target", _GENERIC_TARGETS)
def test_dynamic_source_equals_bucket_source(target):
    # The kernel is runtime-arg, so DYNAMIC and BUCKET emit byte-identical source;
    # only the cache key (shape_key) differs.
    e = get_emitter(target)
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    bucket = e.emit(region, spec=SpecPolicy.BUCKET, dims=(128, 256))
    dyn = e.emit(region, spec=SpecPolicy.DYNAMIC, dims=(128, 256))
    assert dyn.source == bucket.source
    assert dyn.spec is SpecPolicy.DYNAMIC


@pytest.mark.parametrize("target", _GENERIC_TARGETS)
def test_dynamic_shape_key_is_shape_independent(target):
    e = get_emitter(target)
    region = F.FusedRegion(epilogue=("relu",))
    d1 = e.emit(region, spec=SpecPolicy.DYNAMIC, dims=(128, 256))
    d2 = e.emit(region, spec=SpecPolicy.DYNAMIC, dims=(999, 777))
    # DYNAMIC key is the symbolic identity, so it does NOT vary with concrete dims,
    # whereas BUCKET does.
    assert d1.shape_key == d2.shape_key
    b1 = e.emit(region, spec=SpecPolicy.BUCKET, dims=(128, 256))
    b2 = e.emit(region, spec=SpecPolicy.BUCKET, dims=(999, 777))
    assert b1.shape_key != b2.shape_key


@pytest.mark.parametrize("target", _GENERIC_TARGETS)
def test_dynamic_collapses_cache_key_across_shapes(target):
    # The payoff: under DYNAMIC one cache entry serves all shapes; under BUCKET the
    # (source-identical) kernel is keyed — and thus compiled — once per bucket.
    e = get_emitter(target)
    region = F.FusedRegion(epilogue=("bias",))
    kd1 = KC.cache_key(e.emit(region, spec=SpecPolicy.DYNAMIC, dims=(64, 64)),
                       dtype="f32", target=target)
    kd2 = KC.cache_key(e.emit(region, spec=SpecPolicy.DYNAMIC, dims=(4096, 4096)),
                       dtype="f32", target=target)
    assert kd1 == kd2                       # one entry across all shapes
    kb1 = KC.cache_key(e.emit(region, spec=SpecPolicy.BUCKET, dims=(64, 64)),
                       dtype="f32", target=target)
    kb2 = KC.cache_key(e.emit(region, spec=SpecPolicy.BUCKET, dims=(4096, 4096)),
                       dtype="f32", target=target)
    assert kb1 != kb2                       # bucketed: one compile per bucket


@pytest.mark.parametrize("target", _GENERIC_TARGETS)
def test_dynamic_still_honors_dtype_and_region_guards(target):
    # Enabling DYNAMIC must not weaken the other guards (Decision #21).
    e = get_emitter(target)
    with pytest.raises(EmitError, match="f32"):
        e.emit(F.FusedRegion(epilogue=("relu",)), spec=SpecPolicy.DYNAMIC,
               dtype="f16")


# ── 2. Live gfx1151: one DYNAMIC compile serves many shapes ───────────────────

@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="needs a live gfx1151 + hipcc")
def test_live_rocm_dynamic_one_compile_serves_many_shapes():
    from tessera.compiler.emit.rocm_hip import _load_entry, _ptr

    region = F.FusedRegion(epilogue=("bias", "gelu"))
    cache = KC.KernelCache()
    rng = np.random.default_rng(0)
    artifacts = []
    for (M, N, K) in [(64, 64, 64), (256, 128, 96), (512, 512, 512)]:
        compiled = KC.build(region, "rocm", spec=SpecPolicy.DYNAMIC,
                            dtype="f32", dims=(M, N), cache=cache)
        artifacts.append(compiled.artifact)
        fn = _load_entry(compiled.artifact)
        assert fn is not None
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        bias = rng.standard_normal((N,)).astype(np.float32)
        out = np.zeros((M, N), np.float32)
        rc = fn(_ptr(A), _ptr(B), _ptr(bias), _ptr(None), _ptr(out), M, N, K)
        assert rc == 1
        np.testing.assert_allclose(out, region.reference(A, B, bias, None),
                                   rtol=1e-4, atol=1e-4)

    # The whole point: three distinct shapes, ONE compile (one cache entry, one
    # artifact) — the dims-invariant kernel served them all.
    assert cache.misses == 1 and cache.size == 1
    assert len(set(artifacts)) == 1
