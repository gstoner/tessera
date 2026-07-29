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


def _require_toolchain() -> None:
    """Centralized capability gate — `tests/_support/apple.py` owns the probe.

    An inline `skipif` here would be a second, drifting definition of "is the
    Metal toolchain usable"; `test_apple_test_inventory` forbids exactly that.
    """
    from tests._support.apple import require_metal_compiler

    require_metal_compiler()


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


def test_air_build_produces_a_real_metallib_not_a_deferral(tmp_path) -> None:
    _require_toolchain()
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


def test_air_artifact_is_llvm_bitcode_behind_the_container(tmp_path) -> None:
    """`.air` is LLVM bitcode — the property the direct-emission path depends on.

    Checked here so a toolchain change that stopped producing bitcode is caught
    by the AOT lane rather than by whoever later tries to emit AIR directly.
    """
    _require_toolchain()
    region = F.FusedRegion(epilogue=("gelu",))
    source = get_emitter(AIR_TARGET).emit(region, dtype="f16", dims=(64, 64, 64))
    apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                      cache_dir=tmp_path)
    air = next(tmp_path.rglob("k.air"))
    assert air.read_bytes()[:4] == b"\xde\xc0\x17\x0b", "AIR is not LLVM bitcode"


def test_identical_source_compiles_once(tmp_path) -> None:
    _require_toolchain()
    region = F.FusedRegion(epilogue=("gelu",))
    source = get_emitter(AIR_TARGET).emit(region, dtype="f16", dims=(64, 64, 64))
    first = apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                              cache_dir=tmp_path)
    stamp = first.stat().st_mtime_ns
    second = apple_air.compile_msl_to_metallib(source.source, entry=source.entry,
                                               cache_dir=tmp_path)
    assert second == first and second.stat().st_mtime_ns == stamp


def test_bad_msl_raises_naming_the_stage_rather_than_returning_a_path(tmp_path):
    """Decision #21 — a failed compile must not look like a deferral."""
    _require_toolchain()
    with pytest.raises(MetalToolchainError, match="metal -c"):
        apple_air.compile_msl_to_metallib(
            "kernel void broken( { this is not MSL", entry="broken",
            cache_dir=tmp_path)


@pytest.mark.hardware_apple_gpu
def test_aot_and_jit_lanes_are_numerically_identical() -> None:
    """Same MSL, same dispatch body — only library creation differs.

    Bit-identical is the right bar here, not a tolerance: if these two ever
    disagree, the AOT lane is running a different kernel and any AOT-vs-JIT
    timing is measuring the wrong thing.
    """
    _require_toolchain()
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


def test_nonce_control_defeats_metals_shader_cache(tmp_path) -> None:
    """The benchmark's control: a unique nonce must yield a distinct artifact.

    Metal keeps an on-disk shader cache that survives process exit — the same
    kernel measured 140.8 ms in one process and 0.5 ms in the next. Every
    AOT-vs-JIT number depends on each sample being a genuine cache miss. If a
    toolchain change ever made the nonce not affect the artifact, the benchmark
    would quietly start measuring cache hits and report a flattering result, so
    the control is asserted here rather than trusted.
    """
    _require_toolchain()
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_aot_vs_jit",
        Path(__file__).resolve().parents[2]
        / "benchmarks/apple_gpu/benchmark_aot_vs_jit.py")
    assert spec and spec.loader
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    source_a, entry_a = bench._nonce_source("111111")
    source_b, entry_b = bench._nonce_source("222222")
    assert source_a != source_b and entry_a != entry_b

    lib_a = apple_air.compile_msl_to_metallib(source_a, entry=entry_a,
                                              cache_dir=tmp_path)
    lib_b = apple_air.compile_msl_to_metallib(source_b, entry=entry_b,
                                              cache_dir=tmp_path)
    assert lib_a != lib_b, "distinct sources collapsed to one metallib"
    # The bar that matters. An earlier nonce was an *unused* `constant`; the
    # Metal compiler dropped it as dead code and both samples produced
    # byte-identical metallibs, so the AOT lane reloaded one artifact and its
    # measured cost was ~13x too good. Renaming the entry point is what makes
    # each sample a genuinely distinct library.
    assert lib_a.read_bytes() != lib_b.read_bytes(), (
        "distinct sources produced byte-identical metallibs — the nonce is "
        "being optimized away and the benchmark's cache control is broken")


def test_air_runner_is_registered_without_stealing_the_default() -> None:
    """C1: the AOT lane was the only registered target with no runner.

    It must register, and must NOT become the process default — that would move
    every existing F4 verification onto the AOT lane as an import side effect.
    Checked in both import orders because `register_runner` makes the *first*
    registrant the default unless it opts out.
    """
    from tessera.compiler.emit import apple_msl  # noqa: F401 — registers apple_gpu
    from tessera.compiler.emit.kernel_emitter import active_runner, get_runner

    runner = get_runner(AIR_TARGET)
    assert runner.target == AIR_TARGET
    assert active_runner().target == "apple_gpu", (
        "the AOT runner must not be the default; the arbiter selects it")


@pytest.mark.hardware_apple_gpu
def test_air_runner_reports_the_device_tag_only_when_it_dispatched() -> None:
    """The unimplemented methods must decline, not claim a device tag.

    Only `run_fused_region` has an AOT dispatch symbol today. The other three
    have to return a REFERENCE_EXECUTIONS tag so the F4 oracle trusts the
    reference by construction instead of comparing numpy against itself — and
    must not raise, which would stop the oracle gating those regions at all.
    """
    _require_toolchain()
    import numpy as np

    from tessera.compiler.emit.kernel_emitter import (
        REFERENCE_EXECUTIONS, get_runner,
    )

    runner = get_runner(AIR_TARGET)
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)

    _, tag = runner.run_fused_region(F.FusedRegion(epilogue=("gelu",)), a, b)
    assert tag == "metal_metallib" or tag in REFERENCE_EXECUTIONS

    # rank-2 (M, D) — the MSL attention lane unpacks `M, D = Q.shape`.
    q = k = v = (rng.standard_normal((16, 16)) * 0.1).astype(np.float32)
    _, attn_tag = runner.run_fused_attention(F.AttentionRegion(), q, k, v)
    assert attn_tag in REFERENCE_EXECUTIONS, (
        "no AOT attention dispatch exists; claiming a device tag would tell the "
        "F4 oracle to trust a kernel that never ran")


@pytest.mark.hardware_apple_gpu
def test_gpu_executes_hand_written_air_ir(tmp_path) -> None:
    """S0: the GPU runs LLVM IR we wrote, with no MSL front end in the chain.

    Everything else about the AIR path is compile-time evidence — `xcrun metal`
    accepting the IR, `metal-objdump` finding the symbol. Neither shows the
    numbers come out right, and that is the whole question for whether an
    MLIR → AIR emitter is a real direction or a dead end.
    """
    _require_toolchain()
    import ctypes
    import subprocess

    import numpy as np

    from tessera.runtime import _load_apple_gpu_runtime

    ir = Path(__file__).resolve().parents[2] / "tests/data/apple/handwritten_air.ll"
    if not ir.is_file():
        pytest.skip(f"hand-written AIR fixture missing: {ir}")

    air, lib = tmp_path / "hw.air", tmp_path / "hw.metallib"
    for command in (["xcrun", "metal", "-c", str(ir), "-o", str(air)],
                    ["xcrun", "metallib", str(air), "-o", str(lib)]):
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, f"{command[1]} failed:\n{done.stderr}"

    symbol = getattr(_load_apple_gpu_runtime(),
                     "tessera_apple_gpu_metallib_elementwise_f32", None)
    if symbol is None:
        # Not a skip. A dylib without this symbol is stale, not a host lacking a
        # capability — skipping would quietly drop the one test that proves the
        # AIR path executes. Same convention as APPLE-E2E-1.
        raise RuntimeError(
            "libTesseraAppleRuntime is stale: no "
            "tessera_apple_gpu_metallib_elementwise_f32. Rebuild with "
            "`ninja -C build TesseraAppleRuntimeShared`.")
    symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    symbol.restype = ctypes.c_int32

    n = 1024
    x = np.linspace(-3, 3, n).astype(np.float32)
    out = np.zeros(n, np.float32)
    pointer = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ok = symbol(str(lib).encode("utf-8"), b"tessera_handwritten",
                pointer(x), pointer(out), n)
    assert ok, "dispatch of the hand-written AIR kernel declined"
    # The kernel body is `fmul float %v, 3.0` — exact in f32, so demand exact.
    np.testing.assert_array_equal(out, x * 3.0)


@pytest.mark.hardware_apple_gpu
def test_aot_pointwise_matches_the_jit_lane_bitwise() -> None:
    """C2: the second family on the AOT lane.

    Both lanes reach the same `synth_pointwise_impl` in the runtime and share
    `run_pointwise_graph`'s operand preparation — only `acquire_pipeline`
    differs. Bitwise equality is therefore the right bar: any divergence means
    the lanes are dispatching different buffers, not compiling differently.
    """
    _require_toolchain()
    import numpy as np

    from tessera.compiler.emit import apple_msl

    region = F.PointwiseGraphRegion(
        ops=(("mul", ("a", "b"), "t"), ("relu", ("t",), "o")),
        inputs=("a", "b"), output="o")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 32)).astype(np.float32)
    b = rng.standard_normal((8, 32)).astype(np.float32)

    jit_out, jit_tag = apple_msl.run_pointwise_graph(region, [a, b])
    aot_out, aot_tag = apple_air.run_pointwise_graph_aot(region, [a, b])
    if jit_tag != "metal_runtime" or aot_tag == "reference":
        pytest.skip(f"a lane declined the device (jit={jit_tag}, aot={aot_tag})")

    assert aot_tag == "metal_metallib"
    np.testing.assert_array_equal(aot_out, jit_out)
    np.testing.assert_allclose(aot_out, region.reference(a, b), atol=1e-6)


@pytest.mark.hardware_apple_gpu
def test_gpu_executes_hand_written_simdgroup_air_ir(tmp_path) -> None:
    """S1 probe: an emitter can reach the matrix units, not just scalar code.

    `simdgroup_matrix` is Apple's ceiling-setter — it is why SPIR-V was rejected
    as a path. If it were an opaque type or an intrinsic needing backend
    support, an MLIR → AIR emitter would be bounded by whatever we could not
    express. It is neither: `simdgroup_float8x8` is `<64 x float>` and the
    operations are ordinary external functions, so this fixture reaches them
    with plain `declare` + `call`.

    Also exercises the two things a real kernel needs alongside them —
    `addrspace(3)` threadgroup memory and `@air.wg.barrier`.
    """
    _require_toolchain()
    import ctypes
    import subprocess

    import numpy as np

    from tessera.runtime import _load_apple_gpu_runtime

    ir = (Path(__file__).resolve().parents[2]
          / "tests/data/apple/handwritten_air_simdgroup.ll")
    assert ir.is_file(), f"fixture missing: {ir}"

    air, lib = tmp_path / "sg.air", tmp_path / "sg.metallib"
    for command in (["xcrun", "metal", "-c", str(ir), "-o", str(air)],
                    ["xcrun", "metallib", str(air), "-o", str(lib)]):
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, f"{command[1]} failed:\n{done.stderr}"

    symbol = getattr(_load_apple_gpu_runtime(),
                     "tessera_apple_gpu_metallib_elementwise_f32", None)
    if symbol is None:
        raise RuntimeError(
            "libTesseraAppleRuntime is stale: no "
            "tessera_apple_gpu_metallib_elementwise_f32. Rebuild with "
            "`ninja -C build TesseraAppleRuntimeShared`.")
    symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    symbol.restype = ctypes.c_int32

    n = 64  # one 8x8 simdgroup tile
    fill = np.full(n, 7.5, np.float32)
    out = np.zeros(n, np.float32)
    pointer = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ok = symbol(str(lib).encode("utf-8"), b"tessera_sg_probe",
                pointer(fill), pointer(out), n)
    assert ok, "dispatch of the hand-written simdgroup kernel declined"
    np.testing.assert_array_equal(out, np.full(n, 7.5, np.float32))


@pytest.mark.hardware_apple_gpu
def test_aot_pointwise_does_not_depend_on_the_jit_symbol(monkeypatch) -> None:
    """The two lanes must be independently available.

    Caught in review: the shared `run_pointwise_graph` gated entry on the *JIT*
    symbol and then swapped in the AOT one, so a runtime exporting the metallib
    entry but not the synth entry would have silently reported `reference`. The
    lanes are separate arbiter candidates; neither may require the other.
    """
    _require_toolchain()
    import numpy as np

    from tessera.compiler.emit import apple_msl

    region = F.PointwiseGraphRegion(
        ops=(("relu", ("a",), "o"),), inputs=("a",), output="o")
    a = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)

    monkeypatch.setattr(apple_msl, "_synth_pointwise_symbol", lambda elem: None)
    out, tag = apple_air.run_pointwise_graph_aot(region, [a])
    if tag == "reference":
        pytest.skip("device declined the AOT pointwise lane")
    assert tag == "metal_metallib"
    np.testing.assert_allclose(out, region.reference(a), atol=1e-6)


@pytest.mark.hardware_apple_gpu
def test_rebuilt_metallib_at_the_same_path_is_not_served_from_cache(tmp_path):
    """A path is a mutable handle; the pipeline cache must not treat it as an id.

    Tessera names its own artifacts by sha256(source, entry) so they never
    change under a path — but `tessera_apple_gpu_metallib_*` is public C ABI and
    an external caller is under no such obligation. Keyed on path alone, a
    rebuild in place returns the pipeline compiled from the previous bytes.
    """
    _require_toolchain()
    import ctypes
    import subprocess
    import time

    import numpy as np

    from tessera.runtime import _load_apple_gpu_runtime

    template = (Path(__file__).resolve().parents[2]
                / "tests/data/apple/handwritten_air.ll").read_text()
    library = tmp_path / "same_path.metallib"

    def build(multiplier: str) -> None:
        ir = tmp_path / "k.ll"
        ir.write_text(template.replace("fmul float %v, 3.000000e+00",
                                       f"fmul float %v, {multiplier}"))
        for command in (["xcrun", "metal", "-c", str(ir), "-o", str(tmp_path / "k.air")],
                        ["xcrun", "metallib", str(tmp_path / "k.air"),
                         "-o", str(library)]):
            done = subprocess.run(command, capture_output=True, text=True)
            assert done.returncode == 0, done.stderr

    symbol = getattr(_load_apple_gpu_runtime(),
                     "tessera_apple_gpu_metallib_elementwise_f32", None)
    if symbol is None:
        raise RuntimeError("stale libTesseraAppleRuntime — rebuild it")
    symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    symbol.restype = ctypes.c_int32
    pointer = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x = np.full(16, 2.0, np.float32)
    out = np.zeros(16, np.float32)

    build("3.000000e+00")
    assert symbol(str(library).encode(), b"tessera_handwritten",
                  pointer(x), pointer(out), 16)
    assert out[0] == 6.0

    time.sleep(0.02)  # distinct mtime; the key is (path, mtime, size, entry)
    build("5.000000e+00")
    assert symbol(str(library).encode(), b"tessera_handwritten",
                  pointer(x), pointer(out), 16)
    assert out[0] == 10.0, "stale pipeline served for a rebuilt library"


def test_missing_toolchain_declines_instead_of_falling_back(monkeypatch, tmp_path):
    """Never silently produce a JIT result under the AOT target's name."""
    monkeypatch.setattr(apple_air.metal_toolchain_available, "__wrapped__",
                        lambda: False, raising=False)
    monkeypatch.setattr(apple_air, "metal_toolchain_available", lambda: False)
    with pytest.raises(CompileError):
        apple_air.compile_msl_to_metallib("kernel void k() {}", entry="k",
                                          cache_dir=tmp_path)
