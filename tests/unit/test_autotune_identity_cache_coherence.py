"""The identity an arbiter verdict is stamped with names the code that ran.

Sync key ``AUTOTUNE-EMITTED-IDENTITY-2026-09-27`` (Codex review P2 on PR #861).
``NvidiaMmaFusedCandidate.artifact_identity`` hashed the emitter's CURRENT
output while ``run`` launched a function cached by storage/epilogue/raster
alone. Patch or reload the emitter after the lane ran once and a re-measurement
timed the OLD binary under the NEW source's identity; after a restart the
genuinely new binary matched that stamp and reused a latency measured for other
code. The same shape existed in every runtime cache that held a compiled
artifact under a key that did not change with the code.

Every test here drives a lane's real ``run`` (or its loader) with the
compiler, ``dlopen`` and the PTX bridge intercepted, changes the code, and
requires (a) a fresh compile of exactly the new code, (b) the stamp
``autotune._delegate_identities`` writes to equal the digest of the source that
compile received, and (c) unchanged code to keep hitting the cache. Where a
lane deliberately never recompiles within a process (PTX registered once, a
checked-in file compiled once, a library loaded once), the stamp must name the
artifact that is loaded, and a later change must not move it.

Host-independent: no GPU, no nvcc/hipcc. The x86 and checked-in-C++ cases need
the host C/C++ compiler to name its version and skip where there is none.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import shutil
import subprocess

import numpy as np
import pytest

from tessera.compiler import emitted_code_identity as EI
from tessera.compiler import fusion_core as F
from tessera.compiler import toolchain_identity as TI
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import OP_FUSED_REGION, OP_MATMUL, Candidate, Tier

MARK = "AUTOTUNE-EMITTED-IDENTITY cache coherence: emitter changed"


def _changed(fn):
    def wrapper(*a, **k):
        return fn(*a, **k) + f"\n/* {MARK} */\n"
    return wrapper


def _candidate(target, op, name):
    import pkgutil

    import tessera.compiler.emit as emit_pkg

    for mod in pkgutil.iter_modules(emit_pkg.__path__):
        importlib.import_module(f"tessera.compiler.emit.{mod.name}")
    for c in C.candidates_for(target, op):
        if c.name == name:
            return c
    raise AssertionError(f"{name} is not registered for ({target}, {op})")


def _mm(m=32, k=16, n=24):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((m, k)).astype(np.float32),
            rng.standard_normal((k, n)).astype(np.float32),
            rng.standard_normal((n,)).astype(np.float32))


# ── an intercepted toolchain ────────────────────────────────────────────────

class _FakeFn:
    def __init__(self):
        self.restype = None
        self.argtypes = None

    def __call__(self, *a):
        return 1                       # rc=1 for an entry, 1.0 ms for a timer


class _FakeLib:
    def __init__(self, name, mode=None, **_k):
        self._name = str(name)
        self._fns: dict[str, _FakeFn] = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.__dict__["_fns"].setdefault(name, _FakeFn())


class _Compile:
    def __init__(self, argv, text):
        self.argv, self.text = argv, text


_SRC_SUFFIXES = (".cu", ".hip", ".c")


class _Toolchain:
    """Every compile the lanes hand to nvcc/hipcc/cc, as (argv, source text);
    the compile "succeeds" (an output file appears) and `ctypes.CDLL` returns a
    fake library whose entry points report success."""

    def __init__(self, monkeypatch, tmp_path):
        self.compiles: list[_Compile] = []
        self._tmp = tmp_path
        monkeypatch.setattr(subprocess, "run", self._run)
        monkeypatch.setattr(ctypes, "CDLL", _FakeLib)

    def _run(self, argv, *a, **k):
        argv = [str(x) for x in argv]
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "fake cc 1.0\n", "")
        src = next(x for x in argv[1:] if x.endswith(_SRC_SUFFIXES))
        with open(src) as f:
            self.compiles.append(_Compile(argv, f.read()))
        out = argv[argv.index("-o") + 1]
        with open(out, "wb") as f:
            f.write(b"fake shared object %d" % len(self.compiles))
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    from tessera.compiler.emit import kernel_cache as KC
    from tessera.compiler.emit import nvidia_cuda as N
    from tessera.compiler.emit import rocm_hip as R
    from tessera.compiler.emit import x86_c as X

    monkeypatch.setattr(KC, "_DEFAULT_CACHE", KC.KernelCache())
    for attr in ("_mma_fused_fn_cache", "_mma_fused_device_fn_cache",
                 "_mma_attn_fn_cache", "_mma_attn_device_fn_cache",
                 "_mma_gated_fn_cache", "_mma_gated_device_fn_cache",
                 "_EMITTED_ARTIFACTS", "_EMITTED_KEYS", "_LIB_CACHE"):
        monkeypatch.setattr(N, attr, {}, raising=False)
    monkeypatch.setattr(R, "_LIB_CACHE", {})
    monkeypatch.setattr(X, "_LIB_CACHE", {})
    monkeypatch.setattr(EI, "_COMPILER_VERSIONS", {})
    monkeypatch.setattr(TI, "_LOADED", {}, raising=False)
    return _Toolchain(monkeypatch, tmp_path)


def _build_line(argv, name):
    out, skip = [name], False
    for part in argv[1:]:
        if skip:
            skip = False
        elif part == "-o":
            skip = True
        elif not part.endswith(_SRC_SUFFIXES):
            out.append(part)
    return " ".join(out)


def _stamp(cand, region, inputs):
    """What the arbiter stamps for ``cand`` after timing it."""
    stamped = AT._delegate_identities({cand.name: cand}, region, tuple(inputs))
    assert cand.name in stamped, EI.miss_reason(cand.name)
    return stamped[cand.name]


def _assert_names(stamp, compiled, compiler, prefix=""):
    entry = stamp[f"{prefix}entry"]
    assert stamp[f"{prefix}source_sha256"] == EI._units_digest([(entry, compiled.text)]), (
        "the stamp digests different source than the compile that ran")
    assert stamp[f"{prefix}build"] == _build_line(compiled.argv, compiler), (
        "the stamp names different flags than the compile that ran")


# ── NVIDIA emitted mma.sync lanes (the Codex finding) ───────────────────────

_MMA_LANES = [
    # (op, name, region, inputs, emitter, identity prefix, needs ml_dtypes)
    (OP_FUSED_REGION, "nvidia_mma_fused",
     lambda: F.FusedRegion(epilogue=("bias", "relu"), storage_dtype="f16"),
     _mm, "_synthesize_mma_fused_cuda", "", False),
    (OP_FUSED_REGION, "nvidia_mma_fused_fp8_e4m3",
     lambda: F.FusedRegion(epilogue=("bias",), storage_dtype="fp8_e4m3"),
     _mm, "_synthesize_mma_fused_cuda", "", True),
    ("attention", "nvidia_mma_attn_fp8_e4m3",
     lambda: F.AttentionRegion(scale=0.125, storage_dtype="fp8_e4m3"),
     lambda: tuple(np.random.default_rng(2).standard_normal((16, 16))
                   .astype(np.float32) * 0.1 for _ in range(3)),
     "_synthesize_mma_attn_cuda", "mma.", True),
    ("gated_matmul", "nvidia_mma_gated_bf16",
     lambda: F.GatedMatmulRegion(gate_act="silu", storage_dtype="bf16"),
     lambda: (np.ones((16, 32), np.float32), np.ones((32, 16), np.float32),
              np.ones((32, 16), np.float32)),
     "_synthesize_mma_gated_cuda", "", True),
]


@pytest.mark.parametrize("op,name,region,inputs,emitter,prefix,lowp", _MMA_LANES,
                         ids=[lane[1] for lane in _MMA_LANES])
def test_mma_lane_recompiles_a_changed_emitter_and_stamps_what_ran(
        toolchain, monkeypatch, op, name, region, inputs, emitter, prefix, lowp):
    if lowp:
        pytest.importorskip("ml_dtypes")
    from tessera.compiler.emit import nvidia_cuda as N

    cand = _candidate("nvidia", op, name)
    reg, ins = region(), inputs()

    assert cand.run(reg, *ins)[1] == N._REAL_TAG
    assert len(toolchain.compiles) == 1
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[0], "nvcc", prefix)
    # Unchanged code keeps hitting: no recompile on a second run.
    assert cand.run(reg, *ins)[1] == N._REAL_TAG
    assert len(toolchain.compiles) == 1, "unchanged source must not recompile"

    # The emitter changes within the process (a patch, an edit + reload of a
    # helper). The next run must compile the NEW text, and the stamp must be
    # the digest of what that compile received -- not a fresh emission beside
    # an old binary.
    monkeypatch.setattr(N, emitter, _changed(getattr(N, emitter)))
    assert cand.run(reg, *ins)[1] == N._REAL_TAG
    assert len(toolchain.compiles) == 2, "a changed emitter must compile fresh"
    assert MARK in toolchain.compiles[1].text
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[1], "nvcc", prefix)
    # The new source is cached like the old, and the lane's device timer
    # (where it has one) shares that artifact rather than compiling again.
    cand.run(reg, *ins)
    cand.measure_device_latency(reg, *ins)
    assert len(toolchain.compiles) == 2, "the new source is cached like the old"


def test_mma_raster_variants_are_separate_artifacts(toolchain):
    """A non-default raster is a different source; it must not share the
    default raster's artifact (its identity would not describe it)."""
    from tessera.compiler.emit import nvidia_cuda as N

    N._mma_fused_fn(True, "relu", "f16")
    N._mma_fused_fn(True, "relu", "f16", raster_order="grouped_m", raster_group=4)
    N._mma_fused_fn(True, "relu", "f16")
    assert len(toolchain.compiles) == 2
    assert toolchain.compiles[0].text != toolchain.compiles[1].text


def test_resident_stages_recompile_and_the_composed_stamp_names_them(
        toolchain, monkeypatch, tmp_path):
    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda as N

    gemm = tmp_path / "libtessera_nvidia_gemm.so"
    gemm.write_bytes(b"shipped-gemm-v1")
    monkeypatch.setattr(rt, "_nvidia_gemm_lib_path", lambda: gemm)
    monkeypatch.setattr(rt, "_nvidia_gemm_runtime", None)
    dtype_key = next(iter(rt._NVIDIA_GEMM_SYMBOLS))

    def stamp():
        ident = N._composed_identity("eid_composed", dtype_key)
        assert ident is not None, EI.miss_reason("eid_composed")
        return ident

    N._resident_ops_lib()
    N._resident_ops_lib()
    assert len(toolchain.compiles) == 1
    _assert_names(stamp(), toolchain.compiles[0], "nvcc", "stages.")
    monkeypatch.setattr(N, "_synthesize_resident_ops_cuda",
                        _changed(N._synthesize_resident_ops_cuda))
    N._resident_ops_lib()
    assert len(toolchain.compiles) == 2 and MARK in toolchain.compiles[1].text
    _assert_names(stamp(), toolchain.compiles[1], "nvcc", "stages.")


# ── the generic lanes (kernel_cache.build) ──────────────────────────────────

def test_nvidia_generic_lane_recompiles_for_new_source_and_new_arch(
        toolchain, monkeypatch):
    from tessera.compiler.emit import nvidia_cuda as N

    cand = _candidate("nvidia", OP_FUSED_REGION, "nvidia_generic_cuda")
    reg = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32")
    ins = _mm()
    monkeypatch.delenv("TESSERA_NVIDIA_ARCH", raising=False)
    cand.run(reg, *ins)
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 1
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[0], "nvcc")

    monkeypatch.setattr(N, "_synthesize_fused_cuda", _changed(N._synthesize_fused_cuda))
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 2 and MARK in toolchain.compiles[1].text
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[1], "nvcc")

    # The arch is in the identity's build line; it must be in the cache key
    # too, or the artifact built for the old arch is timed under the new one.
    monkeypatch.setenv("TESSERA_NVIDIA_ARCH", "sm_89")
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 3
    assert "-arch=sm_89" in toolchain.compiles[2].argv
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[2], "nvcc")
    monkeypatch.delenv("TESSERA_NVIDIA_ARCH")
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 3, "the default-arch artifact still hits"


def test_rocm_generic_lane_recompiles_for_a_new_offload_arch(toolchain, monkeypatch):
    from tessera.compiler.emit import rocm_hip as R

    cand = _candidate("rocm", OP_FUSED_REGION, "rocm_generic_hip")
    reg, ins = F.FusedRegion(epilogue=("bias", "gelu")), _mm()
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1151")
    cand.run(reg, *ins)
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 1
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[0], "hipcc")
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1201")
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 2
    assert "--offload-arch=gfx1201" in toolchain.compiles[1].argv
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[1], "hipcc")
    monkeypatch.setattr(R, "_synthesize_fused_hip", _changed(R._synthesize_fused_hip))
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 3 and MARK in toolchain.compiles[2].text
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[2], "hipcc")


def test_x86_generic_lane_recompiles_a_changed_emitter(toolchain, monkeypatch):
    from tessera.compiler.emit import x86_c as X

    monkeypatch.setattr(X, "host_supports_x86_64_v4", lambda: True)
    cand = _candidate("x86", OP_FUSED_REGION, "x86_generic_c")
    reg, ins = F.FusedRegion(epilogue=("bias", "gelu")), _mm()
    assert cand.run(reg, *ins)[1] == X._REAL_TAG
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 1
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[0], "cc")
    monkeypatch.setattr(X, "_synthesize_fused_c", _changed(X._synthesize_fused_c))
    cand.run(reg, *ins)
    assert len(toolchain.compiles) == 2 and MARK in toolchain.compiles[1].text
    _assert_names(_stamp(cand, reg, ins), toolchain.compiles[1], "cc")


def test_store_key_is_the_cache_key_without_a_build_line():
    from tessera.compiler.emit import kernel_cache as KC
    from tessera.compiler.emit.kernel_emitter import KernelSource

    src = KernelSource(source="int x;", entry="e", lang="c")
    assert KC.store_key(src, dtype="f32", target="eid_no_line") == KC.cache_key(
        src, dtype="f32", target="eid_no_line")


# ── PTX lanes: registered once, stamped with what the bridge holds ──────────

class _FakeBridge:
    def __init__(self):
        self.held: dict[str, str] = {}
        self.registrations = 0

    def tessera_nvidia_ptx_register(self, name, text):
        self.held[name.decode()] = text.decode()
        self.registrations += 1
        return 0

    def tessera_nvidia_ptx_invoke(self, *a):
        return 0


def test_ptx_lane_stamp_is_the_registered_ptx(monkeypatch, tmp_path):
    from tessera import runtime as rt
    from tessera.compiler import ptx_emit as pe
    from tessera.compiler.gpu_target import ptx_for_driver_jit

    bridge = _FakeBridge()
    lib = tmp_path / "libtessera_nvidia_ptx_launch.so"
    lib.write_bytes(b"bridge-v1")
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib_path", lambda: lib)
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib", None)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", lambda: bridge)
    monkeypatch.setattr(rt, "_nvidia_ptx_registered", {})
    monkeypatch.setattr(TI, "_LOADED", {}, raising=False)

    cand = _candidate("nvidia", OP_MATMUL, "nvidia_mma_gemm_emitted")
    reg = F.MatmulRegion(dtype="float16")
    rng = np.random.default_rng(3)
    ins = (rng.standard_normal((32, 32)).astype(np.float32),
           rng.standard_normal((32, 16)).astype(np.float32))
    entry = pe.MMA_SYNC_GEMM_ENTRY["f16"]

    def stamped_text_is(text):
        assert _stamp(cand, reg, ins)["kernel.source_sha256"] == EI._units_digest(
            [(entry, EI.normalized_ptx(text))])

    assert cand.run(reg, *ins)[1] == "nvidia_ptx_gemm"
    assert bridge.registrations == 1
    registered = rt._nvidia_ptx_registered[entry]
    assert ptx_for_driver_jit(registered)[0] == bridge.held[entry]
    stamped_text_is(registered)

    # The emitter changes. The lane never re-registers an entry, so the bridge
    # keeps running the first PTX -- and the stamp must keep naming it.
    monkeypatch.setattr(pe, "emit_mma_sync_gemm_ptx", _changed(pe.emit_mma_sync_gemm_ptx))
    cand.run(reg, *ins)
    assert bridge.registrations == 1
    stamped_text_is(registered)
    # A fresh process registers the new PTX; its stamp differs, so a verdict
    # stamped for the old PTX misses there.
    rt._nvidia_ptx_registered.clear()
    changed = pe.emit_mma_sync_gemm_ptx(dtype="f16")
    assert MARK in changed
    stamped_text_is(changed)


# ── libraries: identified as the image this process loaded ──────────────────

def test_a_library_rebuilt_after_load_is_a_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(TI, "_LOADED", {}, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", _FakeLib)
    lib = tmp_path / "libdelegate.so"
    lib.write_bytes(b"delegate-v1")
    TI.load_library(lib)
    first = TI.delegate_library_identity(lib)
    assert TI.delegate_library_identity(lib) == first
    rebuilt = tmp_path / "libdelegate.so.tmp"
    rebuilt.write_bytes(b"delegate-v2 (rebuilt)")
    os.replace(rebuilt, lib)          # how a linker lands a rebuilt library
    with pytest.raises(TI.LibraryChangedSinceLoad):
        TI.delegate_library_identity(lib)
    # A library this process never loaded is identified by the file on disk.
    other = tmp_path / "libunloaded.so"
    other.write_bytes(b"unloaded-v1")
    before = TI.delegate_library_identity(other)
    other.write_bytes(b"unloaded-v2 (rebuilt)")
    assert TI.delegate_library_identity(other) != before


def test_ptx_bridge_identity_names_the_loaded_bridge(monkeypatch, tmp_path):
    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda as N

    monkeypatch.setattr(TI, "_LOADED", {}, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", _FakeLib)
    loaded = tmp_path / "a" / "libtessera_nvidia_ptx_launch.so"
    elsewhere = tmp_path / "b" / "libtessera_nvidia_ptx_launch.so"
    for path, body in ((loaded, b"bridge-loaded"), (elsewhere, b"bridge-other")):
        path.parent.mkdir()
        path.write_bytes(body)
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib", TI.load_library(loaded))
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib_path", lambda: elsewhere)
    assert N._ptx_bridge_identity() == TI.delegate_library_identity(loaded)
    loaded.write_bytes(b"bridge-rebuilt-in-place")
    with pytest.raises(TI.LibraryChangedSinceLoad):
        N._ptx_bridge_identity()


# ── checked-in C++ lanes: stamped with the bytes that were compiled ─────────

@pytest.fixture(params=["spectral", "tpp"])
def checked_in_lane(request, monkeypatch, tmp_path):
    if request.param == "spectral":
        mod = importlib.import_module("tessera.compiler.emit.spectral_candidates")
        cand = _candidate("cpu", "spectral_fft", "cpu_stockham")
        region = mod.SpectralFFTRegion(64)
    else:
        mod = importlib.import_module("tessera.compiler.emit.tpp_candidates")
        cand = _candidate("cpu", "tpp_stencil", "cpu_stencil_grad")
        region = mod.StencilGradRegion(8, 8)
    try:
        EI.compiler_version(os.environ.get("CXX", "c++"))
    except EI.EmittedIdentityUnavailable:
        pytest.skip("no host C++ compiler to name")
    hooks = mod._CPU_SRC.parent.parent
    shutil.copytree(hooks, tmp_path / hooks.name)
    src = tmp_path / hooks.name / mod._CPU_SRC.parent.name / mod._CPU_SRC.name
    monkeypatch.setattr(mod, "_CPU_SRC", src)
    compiles: list[str] = []
    edit_during_compile: list[bytes] = []

    def fake_compile_step(out):
        compiles.append(src.read_text())
        if edit_during_compile:
            src.write_bytes(src.read_bytes() + edit_during_compile.pop())
        return _FakeLib(out)

    if request.param == "spectral":
        monkeypatch.setattr(mod, "_libs", {})

        def fake_compile(key, argv, out):
            mod._libs[key] = fake_compile_step(out)
            return mod._libs[key]
        monkeypatch.setattr(mod, "_compile", fake_compile)

        def unload():
            monkeypatch.setattr(mod, "_libs", {})
    else:
        monkeypatch.setattr(mod, "_lib", [])
        monkeypatch.setattr(mod, "_compile_cpu_lib", lambda cxx: fake_compile_step(
            str(tmp_path / f"libtpp_stencil_{len(compiles)}.so")))

        def unload():
            monkeypatch.setattr(mod, "_lib", [])
    return mod, cand, region, src, compiles, unload, edit_during_compile


def test_checked_in_lane_is_stamped_with_the_bytes_it_compiled(checked_in_lane):
    mod, cand, region, src, compiles, unload, _ = checked_in_lane
    on_disk = cand.artifact_identity(region)
    assert on_disk is not None, EI.miss_reason(cand.name)
    assert mod._cpu_lib() is not None and len(compiles) == 1
    assert cand.artifact_identity(region) == on_disk
    # The file is edited while the library stays loaded: the process still
    # runs the old bytes, so the stamp must not move.
    src.write_bytes(src.read_bytes() + b"\n// edited after the load\n")
    assert cand.artifact_identity(region) == on_disk
    mod._cpu_lib()
    assert len(compiles) == 1
    # A fresh process compiles the edited file and stamps it.
    unload()
    fresh = cand.artifact_identity(region)
    assert fresh is not None and fresh != on_disk
    mod._cpu_lib()
    assert len(compiles) == 2 and cand.artifact_identity(region) == fresh


def test_checked_in_lane_edited_during_its_compile_is_a_miss(checked_in_lane):
    mod, cand, region, src, compiles, unload, edit_during_compile = checked_in_lane
    edit_during_compile.append(b"\n// edited while compiling\n")
    assert mod._cpu_lib() is not None
    assert cand.artifact_identity(region) is None
    assert "unknown" in (EI.miss_reason(cand.name) or "")


def test_stockham_identity_covers_the_header_it_includes(checked_in_lane):
    mod, cand, region, src, *_ = checked_in_lane
    if cand.name != "cpu_stockham":
        pytest.skip("the stencil hook includes no local header")
    first = cand.artifact_identity(region)
    assert "FFTPlan.h" in first["units"]
    header = src.parent.parent / "Common" / "FFTPlan.h"
    header.write_bytes(header.read_bytes() + b"\n// header changed\n")
    assert cand.artifact_identity(region) != first


# ── the arbiter never stamps a candidate whose code moved during the race ──

class _Region:
    dtype = "bfloat16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _Versioned(Candidate):
    op = OP_MATMUL
    tier = Tier.SYNTHESIZED

    def __init__(self, name, target, moves):
        self.name, self.target, self.moves, self.version = name, target, moves, 0

    def artifact_identity(self, region, *inputs):
        return {"identity": "test", "version": str(self.version)}

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), "fake_real"

    def measure_device_latency(self, region, *inputs, reps=100, warmup=10):
        if self.moves:
            self.version += 1          # the code changes mid-measurement
        return 1.0 if self.moves else 2.0


def test_an_identity_that_moves_during_the_race_is_not_stamped():
    tgt = "eid_moving_race"
    moving, stable = _Versioned("eid_moving", tgt, True), _Versioned("eid_stable", tgt, False)
    C.register_candidate(moving)
    C.register_candidate(stable)
    try:
        A, B = np.ones((4, 4), np.float32), np.ones((4, 4), np.float32)
        cache = AT.MeasureCache()
        AT.measured_arbitrate(_Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4),
                              dtype="bfloat16", cache=cache, device="fakedev",
                              timing=AT.TIMING_DEVICE, device_repeats=1)
        rec = next(iter(cache._store.values()))
        stamped = rec.evidence.get("delegate_identities") or {}
        assert "eid_stable" in stamped
        assert "eid_moving" not in stamped, (
            "samples that straddle two versions of the code describe neither")
    finally:
        C.unregister_candidate(moving)
        C.unregister_candidate(stable)


# ── an identity's own fields cannot be overwritten ──────────────────────────

@pytest.mark.parametrize("core", ["source_sha256", "build", "entry", "identity",
                                  "normalization", "generator", "lang", "units"])
def test_extra_fields_cannot_replace_core_identity_fields(core):
    with pytest.raises(EI.EmittedIdentityUnavailable, match="collides"):
        EI.source_identity(lang="c", entry="e", units=[("e", "int x;")],
                           build=("cc",), extra={core: "forged"})
    ok = EI.source_identity(lang="c", entry="e", units=[("e", "int x;")],
                            build=("cc",), extra={"cache_key": "k"})
    assert ok["cache_key"] == "k"


def test_composite_parts_cannot_overwrite_each_other():
    with pytest.raises(EI.EmittedIdentityUnavailable, match="twice"):
        EI.composite_identity({"a.b": {"c": "1"}, "a": {"b.c": "2"}})
    out = EI.composite_identity({"a": {"x": "1"}, "b": {"x": "2"}})
    assert out["a.x"] == "1" and out["b.x"] == "2"
