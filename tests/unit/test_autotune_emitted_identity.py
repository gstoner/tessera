"""Every arbiter candidate identifies the code it runs (Decision #11).

Sync key ``AUTOTUNE-EMITTED-IDENTITY-2026-09-27``. Codex review P2 on PR #859:
``Candidate.requires_artifact_identity()`` was ``tier == HAND_TUNED``, so a
SYNTHESIZED or EMITTED candidate with no identity was served on the pin-based
toolchain identity alone -- and when its emitter changed with every pin
unchanged, the arbiter kept serving a corpus winner measured for the OLD
generated kernel. These tests pin the fix:

* every registered candidate requires an identity and computes one from the
  code it would run (no base-class default is left in the registry);
* each Python-emitted lane's identity is deterministic for a representative
  region, is computable with no device and no compiler, and changes when the
  emitted code changes;
* the Codex scenario: a verdict stamped with a generated lane's identity is
  served, the emitter changes with the pins unchanged, and the same lookup
  misses;
* a candidate that cannot establish an identity misses, and cannot opt out.

Host-independent (no GPU, no nvcc/hipcc). The x86 and CPU-file lanes need the
host C/C++ compiler to name its version and skip where there is none.
"""
from __future__ import annotations

import importlib
import pkgutil
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


def _registry():
    import tessera.compiler.emit as emit_pkg

    for mod in pkgutil.iter_modules(emit_pkg.__path__):
        importlib.import_module(f"tessera.compiler.emit.{mod.name}")
    for extra in ("tessera.compiler.native_ann", "tessera.compiler.native_ann_gpu"):
        importlib.import_module(extra)
    return [c for cands in C._CANDIDATES.values() for c in cands
            if type(c).__module__.startswith("tessera.")]


def _candidate(target: str, op: str, name: str) -> Candidate:
    _registry()
    for c in C.candidates_for(target, op):
        if c.name == name:
            return c
    raise AssertionError(f"{name} is not registered for ({target}, {op})")


def _mm(m=32, k=16, n=24, bias=True):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    return (a, b, rng.standard_normal((n,)).astype(np.float32)) if bias else (a, b)


def _changed(fn):
    """A wrapped emitter that returns different code (one added real line)."""
    def wrapper(*a, **k):
        return fn(*a, **k) + "\n/* AUTOTUNE-EMITTED-IDENTITY: emitter changed */\n"
    return wrapper


# ── registry guard ──────────────────────────────────────────────────────────

def test_every_registered_candidate_requires_and_declares_an_identity():
    cands = _registry()
    assert len({c.tier for c in cands}) == 3, "expected all three tiers registered"
    no_opt_out = sorted(c.name for c in cands if not c.requires_artifact_identity())
    assert not no_opt_out, f"candidates opting out of an identity: {no_opt_out}"
    undeclared = sorted(
        c.name for c in cands
        if type(c).artifact_identity is Candidate.artifact_identity
        and type(c).delegate_identity is Candidate.delegate_identity)
    assert not undeclared, (
        "candidates relying on the base-class default (no identity at all): "
        f"{undeclared}")


# ── per-lane: deterministic, host-computable, changes with the code ─────────

def _bridge(tmp_path, monkeypatch, content=b"ptx-launch-bridge-v1"):
    from tessera import runtime as rt

    lib = tmp_path / "libtessera_nvidia_ptx_launch.so"
    lib.write_bytes(content)
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib_path", lambda: lib)
    return lib


def _gemm_lib(tmp_path, monkeypatch, content=b"shipped-gemm-v1"):
    from tessera import runtime as rt

    lib = tmp_path / "libtessera_nvidia_gemm.so"
    lib.write_bytes(content)
    monkeypatch.setattr(rt, "_nvidia_gemm_lib_path", lambda: lib)
    return lib


#: (target, op, name, region, inputs, module, emitter attribute to perturb)
_SOURCE_LANES = [
    ("rocm", OP_FUSED_REGION, "rocm_generic_hip",
     lambda: F.FusedRegion(epilogue=("bias", "gelu")), _mm,
     "tessera.compiler.emit.rocm_hip", "_synthesize_fused_hip"),
    ("nvidia", OP_FUSED_REGION, "nvidia_generic_cuda",
     lambda: F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32"), _mm,
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_fused_cuda"),
    ("nvidia", "attention", "nvidia_flash_attn",
     lambda: F.AttentionRegion(scale=0.125), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_attention_cuda"),
    ("nvidia", "gated_matmul", "nvidia_gated",
     lambda: F.GatedMatmulRegion(gate_act="silu"), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_gated_cuda"),
    ("nvidia", "pointwise", "nvidia_pointwise",
     lambda: F.PointwiseGraphRegion(
         ops=(("mul", ("a", "b"), "t"), ("relu", ("t",), "o")),
         inputs=("a", "b"), output="o"), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_pointwise_cuda"),
    ("nvidia", OP_FUSED_REGION, "nvidia_mma_fused",
     lambda: F.FusedRegion(epilogue=("bias", "relu"), storage_dtype="f16"), _mm,
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_mma_fused_cuda"),
    ("nvidia", "attention", "nvidia_mma_attn_fp8_e4m3",
     lambda: F.AttentionRegion(storage_dtype="fp8_e4m3"), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_mma_attn_cuda"),
    # The scalar lane the mma attention hands large/sharp workloads to is part
    # of that candidate's identity too.
    ("nvidia", "attention", "nvidia_mma_attn",
     lambda: F.AttentionRegion(), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_attention_cuda"),
    ("nvidia", "gated_matmul", "nvidia_mma_gated_bf16",
     lambda: F.GatedMatmulRegion(gate_act="gelu", storage_dtype="bf16"), lambda: (),
     "tessera.compiler.emit.nvidia_cuda", "_synthesize_mma_gated_cuda"),
]


@pytest.mark.parametrize("target,op,name,region,inputs,module,attr", _SOURCE_LANES,
                         ids=[lane[2] + ":" + lane[6] for lane in _SOURCE_LANES])
def test_emitted_source_lane_identity(monkeypatch, target, op, name, region,
                                      inputs, module, attr):
    cand = _candidate(target, op, name)
    reg, ins = region(), inputs()
    # Host-side: no device, no compiler. Nothing may shell out.
    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    first = cand.artifact_identity(reg, *ins)
    assert first is not None, EI.miss_reason(name)
    assert first == cand.artifact_identity(reg, *ins), "identity must be deterministic"
    flat = " ".join(first)
    assert "source_sha256" in flat and "build" in flat
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, attr, _changed(getattr(mod, attr)))
    after = cand.artifact_identity(reg, *ins)
    assert after is not None and after != first, (
        f"{name}: identity did not change when {attr} emitted different code")
    from tessera.compiler.kernel_code_identity import identity_mismatch

    assert any(k.endswith("source_sha256") for k in identity_mismatch(first, after))


def _forbidden(*_a, **_k):
    raise AssertionError("an emitted-source identity must not run a tool")


def test_nvidia_lanes_name_the_arch_they_compile_for(monkeypatch):
    cand = _candidate("nvidia", OP_FUSED_REGION, "nvidia_mma_fused")
    reg = F.FusedRegion(epilogue=("bias",), storage_dtype="f16")
    base = cand.artifact_identity(reg)
    monkeypatch.setenv("TESSERA_NVIDIA_ARCH", "sm_121a")
    other = cand.artifact_identity(reg)
    assert base["build"] != other["build"] and "sm_121a" in other["build"]


def test_rocm_generic_lane_names_the_offload_arch(monkeypatch):
    cand = _candidate("rocm", OP_FUSED_REGION, "rocm_generic_hip")
    reg = F.FusedRegion(epilogue=("bias",))
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1151")
    a = cand.artifact_identity(reg)
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1201")
    b = cand.artifact_identity(reg)
    assert "--offload-arch=gfx1151" in a["build"] and "--offload-arch=gfx1201" in b["build"]
    assert a["cache_key"] == b["cache_key"] and a != b


@pytest.mark.parametrize("name,dtype,producer", [
    ("nvidia_mma_gemm_emitted", "float16", "emit_mma_sync_gemm_ptx"),
    ("nvidia_mma_gemm_emitted", "bfloat16", "emit_mma_sync_gemm_ptx"),
])
def test_ptx_emit_lane_identity(tmp_path, monkeypatch, name, dtype, producer):
    from tessera.compiler import ptx_emit as pe

    cand = _candidate("nvidia", OP_MATMUL, name)
    reg = F.MatmulRegion(dtype=dtype)
    _bridge(tmp_path, monkeypatch)
    first = cand.artifact_identity(reg)
    assert first is not None, EI.miss_reason(name)
    assert first == cand.artifact_identity(reg)
    orig = getattr(pe, producer)
    monkeypatch.setattr(pe, producer, lambda *a, **k: orig(*a, **k) + '\n.pragma "changed";\n')
    assert cand.artifact_identity(reg) != first


def test_ptx_comment_only_change_is_not_a_code_change(tmp_path, monkeypatch):
    """Normalization drops full-line comments (banners), nothing else."""
    from tessera.compiler import ptx_emit as pe

    cand = _candidate("nvidia", OP_MATMUL, "nvidia_mma_gemm_emitted")
    reg = F.MatmulRegion(dtype="float16")
    _bridge(tmp_path, monkeypatch)
    first = cand.artifact_identity(reg)
    orig = pe.emit_mma_sync_gemm_ptx
    monkeypatch.setattr(pe, "emit_mma_sync_gemm_ptx",
                        lambda *a, **k: "// banner v2\n\n" + orig(*a, **k))
    assert cand.artifact_identity(reg) == first


def test_ptx_lanes_carry_the_launch_bridge(tmp_path, monkeypatch):
    from tessera import runtime as rt

    cand = _candidate("nvidia", OP_MATMUL, "nvidia_mma_gemm_emitted")
    reg = F.MatmulRegion(dtype="float16")
    _bridge(tmp_path, monkeypatch)
    first = cand.artifact_identity(reg)
    assert first is not None and "bridge.abi_digest" in first
    _bridge(tmp_path, monkeypatch, content=b"ptx-launch-bridge-v2 rebuilt")
    assert cand.artifact_identity(reg) != first
    monkeypatch.setattr(rt, "_nvidia_ptx_launch_lib_path", lambda: None)
    assert cand.artifact_identity(reg) is None, "no bridge: no identity, a miss"


def test_nvfp4_emitted_lane_identity(tmp_path, monkeypatch):
    from tessera.compiler import ptx_emit as pe
    from tessera.compiler.emit import nvidia_cuda as N

    cand = _candidate("nvidia", N.OP_NVFP4_MATMUL, "nvidia_nvfp4_gemm_emitted")
    reg = N.Nvfp4MatmulRegion(m=16, n=8, k=64)
    _bridge(tmp_path, monkeypatch)
    first = cand.artifact_identity(reg)
    assert first is not None, EI.miss_reason(cand.name)
    orig = pe.emit_nvfp4_gemm_ptx
    monkeypatch.setattr(pe, "emit_nvfp4_gemm_ptx",
                        lambda *a, **k: orig(*a, **k) + '\n.pragma "changed";\n')
    assert cand.artifact_identity(reg) != first


def test_tile_lane_identifies_the_generated_ptx(tmp_path, monkeypatch):
    """`tessera-nvidia-opt`-generated PTX: the identity is the PTX the launch
    registers, so a changed lowering changes it; without the tools there is
    no PTX and the lookup misses."""
    from tessera import runtime as rt

    cand = _candidate("nvidia", OP_MATMUL, "nvidia_tile_matmul_shared")
    reg = F.MatmulRegion(dtype="float16")
    _bridge(tmp_path, monkeypatch)
    entry = "tessera_tile_matmul_shared_f16"
    body = [f".visible .entry {entry}(\n  .param .u64 a\n)\n{{\n  ret;\n}}\n"]
    monkeypatch.setattr(rt, "_nvidia_tile_matmul_ptx",
                        lambda schedule, dtype: (entry, body[0]))
    first = cand.artifact_identity(reg)
    assert first is not None, EI.miss_reason(cand.name)
    assert first["kernel.generator"] == "tessera-nvidia-opt"
    body[0] = body[0].replace("ret;", "bar.sync 0;\n  ret;")
    assert cand.artifact_identity(reg) != first

    def no_tools(schedule, dtype):
        raise RuntimeError("NVIDIA Tile compiler tools unavailable")
    monkeypatch.setattr(rt, "_nvidia_tile_matmul_ptx", no_tools)
    assert cand.artifact_identity(reg) is None
    assert "tools unavailable" in EI.miss_reason(cand.name)


def test_composed_lane_carries_library_and_emitted_stages(tmp_path, monkeypatch):
    from tessera.compiler.emit import nvidia_cuda as N

    cand = _candidate("nvidia", OP_FUSED_REGION, "nvidia_mma_fused_composed_tf32")
    reg = F.FusedRegion(epilogue=("bias", "relu"), storage_dtype="f32")
    _gemm_lib(tmp_path, monkeypatch)
    first = cand.artifact_identity(reg)
    assert first is not None, EI.miss_reason(cand.name)
    assert first["gemm.entry"].endswith("_device")
    assert first == cand.artifact_identity(reg)
    monkeypatch.setattr(N, "_synthesize_resident_ops_cuda",
                        _changed(N._synthesize_resident_ops_cuda))
    stages_changed = cand.artifact_identity(reg)
    assert stages_changed != first
    _gemm_lib(tmp_path, monkeypatch, content=b"shipped-gemm-v2 rebuilt")
    assert cand.artifact_identity(reg) != stages_changed
    from tessera import runtime as rt

    monkeypatch.setattr(rt, "_nvidia_gemm_lib_path", lambda: None)
    assert cand.artifact_identity(reg) is None


def test_x86_generic_c_lane_identity(monkeypatch):
    from tessera.compiler.emit import x86_c

    try:
        EI.compiler_version(x86_c._cc())
    except EI.EmittedIdentityUnavailable:
        pytest.skip("no host C compiler to name")
    cand = _candidate("x86", OP_FUSED_REGION, "x86_generic_c")
    reg = F.FusedRegion(epilogue=("bias", "gelu"))
    ins = _mm()
    first = cand.artifact_identity(reg, *ins)
    assert first is not None, EI.miss_reason(cand.name)
    assert "-march=x86-64-v4" in first["build"] and first["compiler"]
    assert first == cand.artifact_identity(reg, *ins)
    assert cand.artifact_identity(reg) is None, "no operands: no workload, a miss"
    monkeypatch.setattr(x86_c, "_synthesize_fused_c", _changed(x86_c._synthesize_fused_c))
    assert cand.artifact_identity(reg, *ins) != first
    # A different host compiler is a different binary: it misses.
    monkeypatch.setattr(EI, "_COMPILER_VERSIONS",
                        {x86_c._cc(): "clang version 99.0.0 (other)"})
    other = cand.artifact_identity(reg, *ins)
    assert other is not None and other["compiler"] != first["compiler"]


@pytest.mark.parametrize("module,target,op,name,region", [
    ("tessera.compiler.emit.spectral_candidates", "cpu", "spectral_fft",
     "cpu_stockham", lambda m: m.SpectralFFTRegion(64)),
    ("tessera.compiler.emit.tpp_candidates", "cpu", "tpp_stencil",
     "cpu_stencil_grad", lambda m: m.StencilGradRegion(8, 8)),
])
def test_checked_in_source_lane_identity(tmp_path, monkeypatch, module, target, op,
                                         name, region):
    import os

    mod = importlib.import_module(module)
    try:
        EI.compiler_version(os.environ.get("CXX", "c++"))
    except EI.EmittedIdentityUnavailable:
        pytest.skip("no host C++ compiler to name")
    cand = _candidate(target, op, name)
    reg = region(mod)
    first = cand.artifact_identity(reg)
    assert first is not None, EI.miss_reason(name)
    assert first == cand.artifact_identity(reg)
    copy = tmp_path / mod._CPU_SRC.name
    copy.write_bytes(mod._CPU_SRC.read_bytes())
    monkeypatch.setattr(mod, "_CPU_SRC", copy)
    assert cand.artifact_identity(reg) == first, "same bytes, same identity"
    copy.write_bytes(copy.read_bytes() + b"\n// changed\n")
    assert cand.artifact_identity(reg) != first
    copy.unlink()
    assert cand.artifact_identity(reg) is None


def _other_filter_run(self, region, xf, hf, *a, **k):
    return (np.asarray(xf, np.complex64) * np.conj(hf)).astype(np.complex64), "x"


def test_python_lane_identity_follows_its_source(monkeypatch):
    from tessera.compiler.emit import spectral_candidates as SC

    cand = _candidate("cpu", SC.OP_SPECTRAL_FILTER, "cpu_spectral_filter")
    reg = SC.SpectralFilterRegion(33)
    first = cand.artifact_identity(reg)
    assert first is not None and first["identity"] == "python_code"
    assert first == cand.artifact_identity(reg)
    monkeypatch.setattr(SC.SpectralFilterCandidate, "run", _other_filter_run)
    assert cand.artifact_identity(reg) != first


def test_composed_spectral_lane_carries_its_inner_fft():
    from tessera.compiler.emit import spectral_candidates as SC

    inner = _candidate("cpu", SC.OP_SPECTRAL_FFT, "cpu_stockham")
    if not inner.available():
        pytest.skip("the CPU Stockham lane cannot be built on this host")
    cand = _candidate("cpu", SC.OP_SPECTRAL_RFFT, "cpu_rfft")
    ident = cand.artifact_identity(SC.SpectralRFFTRegion(64))
    assert ident is not None, EI.miss_reason(cand.name)
    assert ident["inner_lane.name"] == "cpu_stockham"
    inner_ident = inner.artifact_identity(SC.SpectralFFTRegion(64))
    assert {f"inner.{k}": v for k, v in inner_ident.items()}.items() <= ident.items()


# ── the Codex scenario, end to end ──────────────────────────────────────────

_EID_TARGET = "eid_codex_rocm"


@pytest.fixture
def private_generic_lane():
    """`rocm_generic_hip`'s real identity code under a private target, so the
    live field is exactly this one lane on every host (gfx1151 included)."""
    from tessera.compiler.emit import rocm_hip as R

    cand = R.RocmGenericHipCandidate()
    cand.target = _EID_TARGET
    C.register_candidate(cand)
    yield cand
    C.unregister_candidate(cand)


def test_codex_scenario_changed_emitter_with_unchanged_pins_misses(
        monkeypatch, private_generic_lane):
    from tessera.compiler.emit import rocm_hip as R

    cand = private_generic_lane
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    a, b, bias = _mm()
    dims = AT._infer_dims(OP_FUSED_REGION, (a, b, bias))
    key = ("dev:eid", _EID_TARGET, OP_FUSED_REGION,
           AT.bucket_key(dims, AT.SpecPolicy.BUCKET), "f32", AT.TIMING_END_TO_END)
    cache = AT.MeasureCache()
    stamped = AT._delegate_identities({cand.name: cand}, region, (a, b, bias))
    assert cand.name in stamped
    cache.put(key, AT.MeasureRecord(
        winner=cand.name, latency_ms=1.0, candidates={cand.name: 1.0},
        unmeasured={}, evidence={"delegate_identities": stamped}), fresh=True)
    pins = TI.toolchain_identity(_EID_TARGET).digest

    def ask():
        return AT.corpus_winner(region, OP_FUSED_REGION, _EID_TARGET, a, b, bias,
                                dtype="f32", cache=cache, device="dev:eid")

    assert ask() == cand.name, "the verdict for the code that was timed is served"
    live = C.live_candidates(region, OP_FUSED_REGION, _EID_TARGET, (a, b, bias))
    rec = cache.get(key)
    assert AT._record_matches_live_delegates(rec, live, region, (a, b, bias))

    # The emitter changes; no pin moves.
    monkeypatch.setattr(R, "_synthesize_fused_hip", _changed(R._synthesize_fused_hip))
    TI.clear_identity_cache()
    assert TI.toolchain_identity(_EID_TARGET).digest == pins
    assert cache.get(key) is not None, "the row still loads: only the code moved"
    assert ask() is None, "a verdict for the OLD generated kernel must miss"
    assert not AT._record_matches_live_delegates(rec, live, region, (a, b, bias))


def test_codex_scenario_for_an_emitted_nvidia_lane(monkeypatch):
    """The same failure on the tier the review named: an EMITTED mma.sync lane
    whose emitter changes under unchanged CUDA pins."""
    from tessera.compiler.emit import nvidia_cuda as N

    cand = _candidate("nvidia", OP_FUSED_REGION, "nvidia_mma_fused")
    region = F.FusedRegion(epilogue=("bias", "relu"), storage_dtype="f16")
    ins = _mm()
    rec = AT.MeasureRecord(
        winner=cand.name, latency_ms=1.0, candidates={cand.name: 1.0}, unmeasured={},
        evidence={"delegate_identities": AT._delegate_identities(
            {cand.name: cand}, region, ins)})
    live = {cand.name: cand}
    assert AT._record_matches_live_delegates(rec, live, region, ins)
    monkeypatch.setattr(N, "_synthesize_mma_fused_cuda",
                        _changed(N._synthesize_mma_fused_cuda))
    assert not AT._record_matches_live_delegates(rec, live, region, ins)


# ── a candidate with no identity misses ─────────────────────────────────────

class _Region:
    dtype = "bfloat16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _NoIdentity(Candidate):
    op = OP_MATMUL
    tier = Tier.SYNTHESIZED

    def __init__(self, name, target):
        self.name, self.target = name, target
        self.timed = 0

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), "fake_real"

    def measure_device_latency(self, region, *inputs, reps=100, warmup=10):
        self.timed += 1
        return 1.0


class _OptsOut(_NoIdentity):
    def requires_artifact_identity(self):
        return False


@pytest.mark.parametrize("cls", [_NoIdentity, _OptsOut])
def test_a_candidate_without_an_identity_is_never_served(cls):
    tgt = f"eid_none_{cls.__name__}"
    cand = cls(f"eid_none_{cls.__name__}", tgt)
    C.register_candidate(cand)
    try:
        A, B = _mm(4, 4, 4, bias=False)
        cache = AT.MeasureCache()

        def race():
            return AT.measured_arbitrate(
                _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="bfloat16",
                cache=cache, device="fakedev", timing=AT.TIMING_DEVICE,
                device_repeats=1)

        assert race().name == cand.name
        rec = next(iter(cache._store.values()))
        assert cand.name not in (rec.evidence.get("delegate_identities") or {})
        assert AT.corpus_winner(
            _Region(), OP_MATMUL, tgt, A, B, dims=(4, 4, 4), dtype="bfloat16",
            cache=cache, device="fakedev", timing=AT.TIMING_DEVICE) is None
        timed = cand.timed
        race()
        assert cand.timed > timed, "an unidentified verdict is re-measured, never reused"
    finally:
        C.unregister_candidate(cand)


def test_an_identity_that_raises_is_a_miss_with_a_reason():
    assert EI.identify("eid_raises", lambda: (_ for _ in ()).throw(OSError("gone"))) is None
    assert "OSError: gone" in EI.miss_reason("eid_raises")
    assert EI.identify("eid_empty", lambda: EI.source_identity(
        lang="c", entry="e", units=[("e", "   ")], build=("cc",))) is None
    assert "empty" in EI.miss_reason("eid_empty")
    assert EI.identify("eid_nobuild", lambda: EI.source_identity(
        lang="c", entry="e", units=[("e", "int x;")], build=())) is None


def test_composite_identity_refuses_a_missing_part():
    with pytest.raises(EI.EmittedIdentityUnavailable):
        EI.composite_identity({"a": {"x": "1"}, "b": None})
