"""B4 — the generic synth→compile→cache loop (COMPILER_REFACTOR_PLAN W-B4).

B2 gave every backend an emitter + runner behind one protocol; B3 made the F4
oracle universal. B4 closes the loop with the compile + cache middle steps as an
arch-agnostic driver, with a per-arch ``compile_fn`` plugin seam. These tests
lock:

* ``cache_key`` is content-addressed and separates *source-equal* kernels that
  differ only in dtype / shape-bucket / target / policy;
* ``build`` dedups — a repeated build reuses the compiled kernel (one compile);
* Apple's registered compiler is *deferred* (compile-on-launch, ``artifact is
  None``); a Workstream-C-style ahead-of-time compiler is exercised via a fake;
* the unknown-target diagnostics (Decision #21: name the gap, no silent no-op).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

import tessera.compiler.fusion as F
from tessera.compiler.emit.kernel_cache import (
    CompiledKernel,
    CompileError,
    KernelCache,
    build,
    cache_key,
    get_compiler,
    register_compiler,
)
from tessera.compiler.emit.kernel_emitter import (
    KernelEmitter,
    KernelSource,
    SpecPolicy,
    emit_kernel,
    register_emitter,
)


def _region():
    return F.FusedRegion(epilogue=("gelu",))


# ---- cache_key ---------------------------------------------------------------

def test_cache_key_is_deterministic_and_hex_sha256():
    ks = emit_kernel(_region(), "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16))
    k1 = cache_key(ks, dtype="f32", target="apple_gpu")
    k2 = cache_key(ks, dtype="f32", target="apple_gpu")
    assert k1 == k2
    assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1)


def test_cache_key_separates_dtype_shape_target_policy():
    region = _region()
    base = emit_kernel(region, "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16))
    k = cache_key(base, dtype="f32", target="apple_gpu")
    # dtype
    assert cache_key(base, dtype="f16", target="apple_gpu") != k
    # target
    assert cache_key(base, dtype="f32", target="rocm") != k
    # shape bucket (BUCKET policy quantizes dims -> different shape_key)
    bucketed = emit_kernel(region, "apple_gpu", SpecPolicy.BUCKET, dims=(7, 13, 30))
    assert cache_key(bucketed, dtype="f32", target="apple_gpu") != k


# ---- build loop: dedup + deferred Apple compile ------------------------------

def test_build_dedups_repeated_kernels():
    cache = KernelCache()
    region = _region()
    a = build(region, "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16), cache=cache)
    assert isinstance(a, CompiledKernel)
    assert cache.misses == 1 and cache.hits == 0 and cache.size == 1
    b = build(region, "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16), cache=cache)
    assert b is a and cache.hits == 1 and cache.size == 1


def test_apple_compiler_is_deferred_compile_on_launch():
    cache = KernelCache()
    ck = build(_region(), "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16), cache=cache)
    assert ck.deferred is True and ck.artifact is None
    assert ck.target == "apple_gpu" and ck.entry == ck.source.entry


def test_ahead_of_time_compiler_records_artifact():
    # A Workstream-C-style backend: real ahead-of-time compile returns an artifact.
    # Give it its own emitter (so emit_kernel can produce a source for it) whose
    # source mirrors the Apple one, plus a compiler that "compiles" to a blob.
    class _FakeEmitter(KernelEmitter):
        target = "fake_aot"
        lang = "fake"
        def can_emit(self, region):
            return isinstance(region, F.FusedRegion)
        def emit(self, region, *, spec=SpecPolicy.BUCKET, dtype="f32", dims=None):
            return KernelSource(source="kernel{}", entry="k", lang=self.lang, spec=spec)

    compiled_sources = []

    def _fake_compile(src):
        compiled_sources.append(src.source)
        return b"ARTIFACT-BLOB"

    register_emitter(_FakeEmitter())
    register_compiler("fake_aot", _fake_compile)
    cache = KernelCache()
    ck = build(_region(), "fake_aot", cache=cache)
    assert ck.deferred is False and ck.artifact == b"ARTIFACT-BLOB"
    assert compiled_sources == ["kernel{}"]
    # second build hits the cache — compile fn is NOT called again
    build(_region(), "fake_aot", cache=cache)
    assert compiled_sources == ["kernel{}"] and cache.hits == 1


# ---- diagnostics -------------------------------------------------------------

def test_unknown_compiler_target_raises():
    with pytest.raises(CompileError, match="no compiler registered"):
        get_compiler("nvidia")


def test_get_compiler_bootstraps_apple_without_prior_import():
    # Cold path: probe the compiler seam without importing the facade / apple_msl
    # first. get_compiler must bootstrap the Apple reference compiler so the
    # advertised backend is available regardless of import order (mirrors the
    # emitter registry).
    code = (
        "from tessera.compiler.emit.kernel_cache import get_compiler\n"
        "import tessera.compiler.emit.kernel_cache as kc\n"
        "assert 'apple_gpu' not in kc._COMPILERS, 'apple must not be pre-registered'\n"
        "fn = get_compiler('apple_gpu')\n"
        "assert callable(fn) and fn(object()) is None  # deferred compile-on-launch\n"
        "print('BOOTSTRAP_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(_REPO), env={**os.environ, "PYTHONPATH": "python"})
    assert r.returncode == 0, r.stderr
    assert "BOOTSTRAP_OK" in r.stdout


def test_register_compiler_requires_target():
    with pytest.raises(ValueError, match="non-empty backend id"):
        register_compiler("", lambda s: None)


def test_build_surfaces_compile_failure():
    def _boom(src):
        raise RuntimeError("toolchain exploded")

    class _E(KernelEmitter):
        target = "boom_backend"
        lang = "x"
        def can_emit(self, region):
            return True
        def emit(self, region, *, spec=SpecPolicy.BUCKET, dtype="f32", dims=None):
            return KernelSource(source="s", entry="e", lang=self.lang, spec=spec)

    register_emitter(_E())
    register_compiler("boom_backend", _boom)
    with pytest.raises(CompileError, match="toolchain exploded"):
        build(_region(), "boom_backend", cache=KernelCache())
