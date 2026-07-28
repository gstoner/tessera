"""The ahead-of-time Apple GPU target: MSL → AIR → ``.metallib``.

`apple_gpu` compiles on launch (`compile_fn` returns None). `apple_gpu_air`
compiles through Apple's Metal toolchain and hands back a real artifact, so the
two are comparable candidates rather than a build-time switch.

Structural tests run anywhere. The compile tests need the downloadable Metal
toolchain (`xcodebuild -downloadComponent MetalToolchain`) and skip without it —
they must never fall back to the JIT lane, or an "AOT" measurement would quietly
be a JIT one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tessera.compiler.fusion as F
from tessera.compiler.emit import apple_air
from tessera.compiler.emit.apple_air import AIR_TARGET, MetalToolchainError
from tessera.compiler.emit.kernel_cache import (
    CompileError, KernelCache, build, get_compiler,
)
from tessera.compiler.emit.kernel_emitter import SpecPolicy, get_emitter


requires_toolchain = pytest.mark.skipif(
    not apple_air.metal_toolchain_available(),
    reason="Apple Metal toolchain unavailable (xcodebuild -downloadComponent "
           "MetalToolchain)",
)


def test_air_target_registers_its_own_emitter_and_compiler() -> None:
    assert get_emitter(AIR_TARGET).target == AIR_TARGET
    assert get_compiler(AIR_TARGET) is not None
    # Distinct from the JIT lane — the arbiter has to be able to hold both.
    assert get_compiler("apple_gpu") is not get_compiler(AIR_TARGET)


def test_air_emitter_delegates_so_both_lanes_compile_identical_msl() -> None:
    """Two emitters would drift, and then AOT-vs-JIT would compare kernels."""
    region = F.FusedRegion(epilogue=("gelu",))
    air = get_emitter(AIR_TARGET).emit(region, dtype="f16", dims=(64, 64, 64))
    msl = get_emitter("apple_gpu").emit(region, dtype="f16", dims=(64, 64, 64))
    assert air.source == msl.source
    assert air.entry == msl.entry and air.lang == msl.lang == "msl"


def test_toolchain_hint_is_actionable() -> None:
    hint = apple_air.toolchain_hint()
    assert "MetalToolchain" in hint or "Xcode" in hint


@requires_toolchain
def test_air_build_produces_a_real_metallib_not_a_deferral(tmp_path) -> None:
    cache = KernelCache()
    region = F.FusedRegion(epilogue=("gelu",))
    built = build(region, AIR_TARGET, SpecPolicy.BUCKET, dtype="f16",
                  dims=(64, 64, 64), cache=cache)

    assert built.deferred is False, "AOT target must not report compile-on-launch"
    artifact = Path(built.artifact)
    assert artifact.is_file() and artifact.suffix == ".metallib"
    assert artifact.stat().st_size > 0

    # The JIT lane still defers, and keys separately.
    jit = build(region, "apple_gpu", SpecPolicy.BUCKET, dtype="f16",
                dims=(64, 64, 64), cache=cache)
    assert jit.deferred is True and jit.artifact is None
    assert jit.key != built.key


@requires_toolchain
def test_air_artifact_is_llvm_bitcode_behind_the_container(tmp_path) -> None:
    """`.air` is LLVM bitcode — the property the direct-emission path depends on.

    Checked here so a toolchain change that stopped producing bitcode is caught
    by the AOT lane rather than by whoever later tries to emit AIR directly.
    """
    region = F.FusedRegion(epilogue=("gelu",))
    source = get_emitter(AIR_TARGET).emit(region, dtype="f16", dims=(64, 64, 64))
    apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                      cache_dir=tmp_path)
    air = next(tmp_path.rglob("k.air"))
    assert air.read_bytes()[:4] == b"\xde\xc0\x17\x0b", "AIR is not LLVM bitcode"


@requires_toolchain
def test_identical_source_compiles_once(tmp_path) -> None:
    region = F.FusedRegion(epilogue=("gelu",))
    source = get_emitter(AIR_TARGET).emit(region, dtype="f16", dims=(64, 64, 64))
    first = apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                              cache_dir=tmp_path)
    stamp = first.stat().st_mtime_ns
    second = apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                               cache_dir=tmp_path)
    assert second == first and second.stat().st_mtime_ns == stamp


@requires_toolchain
def test_bad_msl_raises_naming_the_stage_rather_than_returning_a_path(tmp_path):
    """Decision #21 — a failed compile must not look like a deferral."""
    with pytest.raises(MetalToolchainError, match="metal -c"):
        apple_air.compile_msl_to_metallib(
            "kernel void broken( { this is not MSL", entry="broken",
            cache_dir=tmp_path)


@requires_toolchain
@pytest.mark.hardware_apple_gpu
def test_aot_and_jit_lanes_are_numerically_identical() -> None:
    """Same MSL, same dispatch body — only library creation differs.

    Bit-identical is the right bar here, not a tolerance: if these two ever
    disagree, the AOT lane is running a different kernel and any AOT-vs-JIT
    timing is measuring the wrong thing.
    """
    import numpy as np

    from tessera.compiler.emit import apple_msl

    region = F.FusedRegion(epilogue=("gelu",))
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((128, 128)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((128, 128)) * 0.1).astype(np.float16)

    aot, aot_tag = apple_air.run_fused_region_aot(region, a, b)
    jit, jit_tag = apple_msl.run_fused_region(region, a, b)
    if aot_tag == "reference" or jit_tag != "metal_runtime":
        pytest.skip(f"a lane declined the device (aot={aot_tag}, jit={jit_tag})")

    assert aot_tag == "metal_metallib", "AOT tag must name the metallib lane"
    np.testing.assert_array_equal(np.asarray(aot, np.float32),
                                  np.asarray(jit, np.float32))
    reference = region.reference(np.asarray(a, np.float32),
                                 np.asarray(b, np.float32), None, None)
    np.testing.assert_allclose(np.asarray(aot, np.float32), reference,
                               atol=1e-3, rtol=1e-3)


def test_aot_runner_reports_reference_when_the_device_declines(monkeypatch) -> None:
    """Never label a numpy result as the metallib lane."""
    import numpy as np

    monkeypatch.setattr(apple_air, "_metallib_coopmat_symbol", lambda: None)
    region = F.FusedRegion(epilogue=("gelu",))
    a = np.ones((32, 32), np.float16)
    b = np.ones((32, 32), np.float16)
    out, tag = apple_air.run_fused_region_aot(region, a, b)
    assert tag == "reference"
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        region.reference(np.asarray(a, np.float32), np.asarray(b, np.float32),
                         None, None),
        atol=1e-2, rtol=1e-2)


def test_missing_toolchain_declines_instead_of_falling_back(monkeypatch, tmp_path):
    """Never silently produce a JIT result under the AOT target's name."""
    monkeypatch.setattr(apple_air.metal_toolchain_available, "__wrapped__",
                        lambda: False, raising=False)
    monkeypatch.setattr(apple_air, "metal_toolchain_available", lambda: False)
    with pytest.raises(CompileError):
        apple_air.compile_msl_to_metallib("kernel void k() {}", entry="k",
                                          cache_dir=tmp_path)
