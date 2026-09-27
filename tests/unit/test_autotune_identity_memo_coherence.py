"""Identity memos cannot outlive the code they describe; unchanged code is cheap.

Sync ``AUTOTUNE-EMITTED-IDENTITY-2026-09-27`` follow-ups on PR #861.

**A -- ``AUTOTUNE-KERNEL-IDENTITY-MEMO``.** ``kernel_code_identity`` memoized a
``tessera-opt`` image's identity under the lane's selectors plus the
``tessera-opt`` digest, while the runtime hsaco caches key on the directive
text the Python generator produces. A generator change within one process
launched a new image under the memoized OLD identity. The memo is now reused
only for a byte-identical image, and the image is the one the launch's own
build path returns. These tests drive the real ``_rocm_wmma_fused_image`` /
``_rocm_flash_attn_image`` (with ``tessera-opt`` and the disassembler
intercepted), change the directive generator in-process, and require a fresh
image and an identity that names it.

**B -- hot path.** Keying the emitted lanes' artifact caches by their source
made every launch re-run the Python emitter (~10 us) to find the key.
``source_memo`` memoizes the source against the emitter function *object* and
everything it reaches by name. These tests count emitter calls: an unchanged
emitter runs once; a patched emitter -- or a patched helper it calls -- runs
again (the coherence tests in ``test_autotune_identity_cache_coherence.py``
cover the compile-and-stamp half of that and run unchanged).

Host-independent: no GPU, no ``tessera-opt``, no nvcc.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess

import numpy as np
import pytest

from tessera.compiler import fusion_core as F
from tessera.compiler import kernel_code_identity as KI
from tessera.compiler.emit import source_memo as SM

MARK = "AUTOTUNE-KERNEL-IDENTITY-MEMO: directive generator changed"


# ── A: tessera-opt images ──────────────────────────────────────────────────

class _Opt:
    """``tessera-opt`` intercepted: each directive "compiles" to an ELF whose
    bytes are a digest of the directive text, so a changed directive is a
    changed image and an unchanged one is byte-identical."""

    def __init__(self, monkeypatch):
        from tessera import runtime as rt

        self.directives: list[str] = []
        self._real_run = subprocess.run
        monkeypatch.setattr(subprocess, "run", self._run)
        monkeypatch.setattr(rt, "_tessera_opt_path", lambda: "/fake/tessera-opt")
        monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1151")
        monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1151")
        monkeypatch.setattr(rt, "_rocm_compiled_hsaco_cache", {})
        monkeypatch.setattr(rt, "_rocm_fa_hsaco_cache", {})
        monkeypatch.setattr(KI, "generator_fingerprint", lambda: "sha256:opt")
        monkeypatch.setattr(KI, "find_llvm_objdump", lambda: "/fake/llvm-objdump")
        self.disassembled: list[bytes] = []

        def disasm(payload, *, entry_symbol, isa, objdump=None):
            self.disassembled.append(payload)
            return {"identity": "kernel_code", "isa": isa, "entry": entry_symbol,
                    "instruction_stream_sha256": hashlib.sha256(payload).hexdigest()}

        monkeypatch.setattr(KI, "hsaco_kernel_identity", disasm)
        KI.clear_kernel_identity_cache()

    @staticmethod
    def image_of(directive: str) -> bytes:
        return b"\x7fELF" + hashlib.sha256(directive.encode()).hexdigest().encode()

    def _run(self, argv, *a, **k):
        argv = [str(x) for x in argv]
        if not argv or argv[0] != "/fake/tessera-opt":
            return self._real_run(argv, *a, **k)
        directive = k["input"]
        self.directives.append(directive)
        blob = "".join(f"\\{b:02X}" for b in self.image_of(directive))
        return subprocess.CompletedProcess(
            argv, 0, f'gpu.binary @k [#gpu.object<#rocdl.target, bin = "{blob}">]\n', "")


@pytest.fixture
def opt(monkeypatch):
    yield _Opt(monkeypatch)
    KI.clear_kernel_identity_cache()


def _wmma_identity(m=64, n=48, k=32):
    from tessera.compiler.emit import rocm_hip

    cand = rocm_hip.RocmWmmaGemmCandidate.__new__(rocm_hip.RocmWmmaGemmCandidate)
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    ident = cand.artifact_identity(region, np.zeros((m, k), np.float32),
                                   np.zeros((k, n), np.float32), np.zeros(n, np.float32))
    assert ident is not None
    return ident


def _launched_wmma_image(m=64, n=48, k=32):
    """The image `_rocm_wmma_fused_2d` loads for this workload on gfx11 (the
    launch's own selection; the HIP launch itself is not reachable here)."""
    from tessera import runtime as rt

    hsaco, _ = rt._rocm_wmma_fused_gfx11_image(m, n, k, "f16", bias=True,
                                               activation="gelu")
    return hsaco


def _names(identity, image):
    return identity["instruction_stream_sha256"] == hashlib.sha256(image).hexdigest()


def test_wmma_identity_follows_an_in_process_directive_change(opt, monkeypatch):
    from tessera.compiler import rocm_schedule

    first = _wmma_identity()
    assert len(opt.directives) == 1
    assert _names(first, _launched_wmma_image())
    # Unchanged generator: no rebuild, no re-disassembly, the same identity.
    for _ in range(3):
        assert _wmma_identity() == first
    assert len(opt.directives) == 1 and len(opt.disassembled) == 1

    # The directive generator changes within the process: the schedule it
    # projects into the directive now carries another pipeline depth.
    real = rocm_schedule.select_rocm_gemm_schedule

    def deeper(*a, **k):
        s = real(*a, **k)
        return dataclasses.replace(s, pipeline_stages=s.pipeline_stages + 1)

    monkeypatch.setattr(rocm_schedule, "select_rocm_gemm_schedule", deeper)
    launched = _launched_wmma_image()
    assert len(opt.directives) == 2, "a changed directive must build a fresh image"
    second = _wmma_identity()
    assert second != first, "the memoized identity of the OLD image was served"
    assert _names(second, launched), "the identity does not name the image the launch runs"
    assert len(opt.directives) == 2, "identity and launch must share one image"
    assert _wmma_identity() == second and len(opt.disassembled) == 2


def test_flash_attn_identity_follows_an_in_process_directive_change(opt, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.emit import rocm_hip

    class _Attn:
        def _natural(self, Q, K):
            return np.asarray(Q), np.asarray(K)

    fa = rocm_hip.RocmFlashAttnCandidate()
    q = np.zeros((16, 64), np.float32)

    def ident():
        got = fa.artifact_identity(_Attn(), q, q, q)
        assert got is not None
        return got

    first = ident()
    assert _names(first, rt._rocm_flash_attn_image(64, "f16")[0])
    assert ident() == first and len(opt.directives) == 1
    # The variant selection the directive is generated from changes in-process
    # (head_dim 64 now selects the two-wave kernel).
    monkeypatch.setattr(rt, "_rocm_flash_attn_two_wave", lambda *a, **k: True)
    launched, entry = rt._rocm_flash_attn_image(64, "f16")
    assert "two_wave = true" in opt.directives[-1] and len(opt.directives) == 2
    second = ident()
    assert second != first and _names(second, launched) and second["entry"] == entry
    assert len(opt.directives) == 2


def test_a_failed_identity_stays_a_miss_and_never_serves(opt):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("no tessera-opt")

    key = ("eid-memo", "broken")
    assert KI.compiler_kernel_identity(key, broken, isa="gfx1151") is None
    assert KI.compiler_kernel_identity(key, broken, isa="gfx1151") is None
    assert len(calls) == 1 and "no tessera-opt" in KI.miss_reason(key)


# ── B: emitted source memo ──────────────────────────────────────────────────

def _counting(fn):
    calls = []

    def wrapper(*a, **k):
        calls.append(1)
        return fn(*a, **k)
    return wrapper, calls


_SOURCES = [
    # (emitter to count, source getter)
    ("_synthesize_mma_fused_cuda", lambda N: N._mma_fused_source(True, "relu", "f16")),
    ("_synthesize_mma_attn_cuda", lambda N: N._mma_attn_source("fp8_e4m3")),
    ("_synthesize_mma_gated_cuda", lambda N: N._mma_gated_source("bf16", "silu")),
    ("_synthesize_resident_ops_cuda", lambda N: N._resident_ops_source()),
]


@pytest.mark.parametrize("emitter,get", _SOURCES, ids=[s[0] for s in _SOURCES])
def test_an_unchanged_emitter_is_not_rerun(monkeypatch, emitter, get):
    from tessera.compiler.emit import nvidia_cuda as N

    wrapped, calls = _counting(getattr(N, emitter))
    monkeypatch.setattr(N, emitter, wrapped)
    first = get(N)
    for _ in range(10):
        assert get(N) is first, "identity and launch must read one source object"
    assert len(calls) == 1, "an unchanged emitter must not be re-run per launch"
    # Replacing the function object is a change: the memo misses.
    again, calls2 = _counting(wrapped)
    monkeypatch.setattr(N, emitter, again)
    assert get(N).source == first.source and len(calls2) == 1
    get(N)
    assert len(calls2) == 1


def test_a_patched_helper_of_the_emitter_is_seen(monkeypatch):
    """The memo checks every global the emitter reaches by name, not only the
    emitter: patching a helper re-emits."""
    from tessera.compiler.emit import nvidia_cuda as N

    before = N._mma_fused_source(True, "relu", "f16")
    helper = N._native_mma_word_loader
    monkeypatch.setattr(N, "_native_mma_word_loader",
                        lambda *a, **k: helper(*a, **k) + "\n/* helper changed */\n")
    after = N._mma_fused_source(True, "relu", "f16")
    assert "helper changed" in after.source and after is not before


def test_mma_launch_does_not_resynthesize(monkeypatch, tmp_path):
    """The launch path end to end (compile and dlopen intercepted): ten
    `_mma_fused_fn` calls run the emitter once and compile once."""
    from tessera.compiler.emit import nvidia_cuda as N

    compiles = []

    def compile_fn(src):
        compiles.append(src.source)
        out = tmp_path / f"k{len(compiles)}.so"
        out.write_bytes(b"x")
        return str(out)

    class _Lib:
        def __getattr__(self, name):
            return type("F", (), {"restype": None, "argtypes": None})()

    monkeypatch.setattr(N, "_nvidia_cuda_compile_fn", compile_fn)
    monkeypatch.setattr(N, "_load_lib", lambda path: _Lib())
    monkeypatch.setattr(N, "_EMITTED_ARTIFACTS", {})
    monkeypatch.setattr(N, "_mma_fused_fn_cache", {})
    wrapped, calls = _counting(N._synthesize_mma_fused_cuda)
    monkeypatch.setattr(N, "_synthesize_mma_fused_cuda", wrapped)
    for _ in range(10):
        N._mma_fused_fn(False, "gelu", "f16")
    assert len(calls) == 1 and len(compiles) == 1


def test_generic_emit_kernel_is_memoized_and_sees_a_patch(monkeypatch):
    from tessera.compiler.emit import nvidia_cuda as N
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, emit_kernel

    region = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32")
    wrapped, calls = _counting(N._synthesize_fused_cuda)
    monkeypatch.setattr(N, "_synthesize_fused_cuda", wrapped)
    first = emit_kernel(region, "nvidia", SpecPolicy.BUCKET, dtype="f32", dims=None)
    assert emit_kernel(region, "nvidia", SpecPolicy.BUCKET, dtype="f32", dims=None) is first
    assert len(calls) == 1
    # The emitter method itself, patched on the class, is a change too.
    real_emit = N.NvidiaCudaEmitter.emit

    def emit(self, *a, **k):
        src = real_emit(self, *a, **k)
        return dataclasses.replace(src, source=src.source + "\n/* emit patched */\n")

    monkeypatch.setattr(N.NvidiaCudaEmitter, "emit", emit)
    assert "emit patched" in emit_kernel(region, "nvidia", SpecPolicy.BUCKET,
                                         dtype="f32", dims=None).source


# ── the memo's own rules ────────────────────────────────────────────────────

def test_an_environment_reading_emitter_is_never_memoized():
    import os

    calls = []

    def reads_env(x):
        calls.append(1)
        return f"{x}:{os.environ.get('EID_MEMO_PROBE', '')}"

    ns = {"reads_env": reads_env}
    SM.memoized(ns, "reads_env", 1)
    SM.memoized(ns, "reads_env", 1)
    assert len(calls) == 2


def test_unhashable_arguments_are_called_every_time():
    calls = []

    def emit(xs):
        calls.append(1)
        return str(xs)

    ns = {"emit": emit}
    SM.memoized(ns, "emit", [1, 2])
    SM.memoized(ns, "emit", [1, 2])
    assert len(calls) == 2


def test_stateful_or_opted_out_objects_are_called_every_time():
    calls = []

    class Stateful:
        def __init__(self):
            self.tile = 4

        def emit(self, x):
            calls.append(1)
            return x * self.tile

    class Delegating:
        source_memo_safe = False

        def emit(self, x):
            calls.append(1)
            return x

    for obj in (Stateful(), Delegating()):
        calls.clear()
        SM.memoized_method(obj, "emit", 3)
        SM.memoized_method(obj, "emit", 3)
        assert len(calls) == 2, type(obj).__name__


def test_a_rebinding_during_the_emit_is_not_memoized():
    """If the code the emitter reaches is rebound while it runs, which code
    produced the value is unknown: it is returned but not memoized."""
    calls: list[int] = []
    ns: dict = {"_calls": calls}
    exec(  # noqa: S102 - a namespace whose globals the walk reads
        "def helper():\n"
        "    return 'a'\n"
        "def emit():\n"
        "    global helper\n"
        "    _calls.append(1)\n"
        "    helper = lambda: 'b'\n"
        "    return helper()\n", ns)
    SM.memoized(ns, "emit")
    SM.memoized(ns, "emit")
    assert len(calls) == 2


def test_numerically_equal_arguments_of_different_types_are_not_aliased():
    """`1 == 1.0 == True` hash alike, but an emitter formatting them into C
    writes different programs (`a / 2` vs `a / 2.0`); the memo key tells them
    apart, including inside a frozen region."""
    ns: dict = {}
    exec("def emit(d):\n    return f'x / {d}'\n", ns)  # noqa: S102
    assert [SM.memoized(ns, "emit", d) for d in (2, 2.0, True, 1)] == [
        "x / 2", "x / 2.0", "x / True", "x / 1"]
    exec("def emit_region(r):\n    return f'scale={r.scale}'\n", ns)  # noqa: S102
    assert SM.memoized(ns, "emit_region", F.AttentionRegion(scale=1)) == "scale=1"
    assert SM.memoized(ns, "emit_region", F.AttentionRegion(scale=1.0)) == "scale=1.0"


def test_a_mutable_argument_is_never_memoized():
    calls: list[int] = []
    ns: dict = {"_calls": calls}
    exec("def emit(o):\n    _calls.append(1)\n    return str(o.v)\n", ns)  # noqa: S102

    class Box:
        v = 1

    b = Box()
    assert SM.memoized(ns, "emit", b) == "1"
    b.v = 2
    assert SM.memoized(ns, "emit", b) == "2" and len(calls) == 2
