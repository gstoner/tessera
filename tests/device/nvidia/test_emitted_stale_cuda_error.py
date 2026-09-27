"""A stale CUDA error must not fail an emitted sm_120 lane.

Sync ``SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`` (the emitted-template half of
``SPECTRAL-STALE-HIP-ERROR-2026-09-27``). The CUDA runtime keeps one
last-error slot per thread and per runtime instance, and only
``cudaGetLastError()`` resets it. Every library ``nvidia_cuda`` compiles links
cudart **statically** (nvcc's default; its cudart symbols are local), so each
emitted .so owns a private runtime instance -- priming the process's shared
``libcudart`` would not touch the slot these libraries read and would prove
nothing. The prime therefore goes through the library's OWN local
``cudaSetDevice`` (resolved from its symbol table and load base) on an ordinal
that does not exist, confirmed by its own ``cudaPeekAtLastError`` (which does
not reset the slot) -- the state an earlier refused call into the same
library leaves behind.

For every emitted source that reads the slot, each test:

1. **negative control** -- runs the lane with the entry clears stripped from
   the emitter (a different source, so a different artifact) under a primed
   slot and requires it to FAIL: the prime demonstrably reaches the slot the
   lane reads, so the positive half cannot pass by not priming anything.
   Exception, recorded per lane: the two mma.sync attention entries call
   ``cudaFuncSetAttribute`` before launching, and a successful call RESETS the
   slot (measured on this box), so their stripped variants are masked;
2. runs the shipped emitter under a primed slot and requires the real kernel
   to run, agree with the reference, and leave the slot clean.

Device timers are exercised the same way (``measure_device_latency`` must
return a latency, not ``None``).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = pytest.mark.hardware_nvidia

_MISSING_ORDINAL = 97
_CLEAR = "(void)cudaGetLastError();"


def _require_sm120():
    from tessera import runtime as rt

    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("exact sm_120 device is unavailable")
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("NVIDIA mma runtime is unavailable")


@dataclass
class _PrivateRuntime:
    """The local cudart entry points of one statically linked emitted .so."""

    set_device: Callable[[int], int]
    peek: Callable[[], int]
    get_last: Callable[[], int]
    device_count: Callable[[Any], int]


def _load_base(path: str) -> int:
    real = os.path.realpath(path)
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            parts = line.split()
            if len(parts) >= 6 and parts[-1] == real and int(parts[2], 16) == 0:
                return int(parts[0].split("-")[0], 16)
    raise AssertionError(f"{real} is not mapped in this process")


def _private_runtime(path: str) -> _PrivateRuntime:
    dynamic = subprocess.run(["nm", "-D", "--defined-only", path], check=True,
                             capture_output=True, text=True).stdout
    assert " cudaGetLastError" not in dynamic, (
        f"{path} exports cudart: its runtime instance is not private, so this "
        "test's premise (and priming method) no longer holds")
    table = subprocess.run(["nm", path], check=True, capture_output=True,
                           text=True).stdout
    wanted = ("cudaSetDevice", "cudaPeekAtLastError", "cudaGetLastError",
              "cudaGetDeviceCount")
    found: dict[str, list[int]] = {}
    for line in table.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("t", "T") and parts[2] in wanted:
            found.setdefault(parts[2], []).append(int(parts[0], 16))
    assert all(len(found.get(name, ())) == 1 for name in wanted), found
    base = _load_base(path)

    def fn(name: str, *argtypes: Any) -> Any:
        return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(base + found[name][0])

    return _PrivateRuntime(
        set_device=fn("cudaSetDevice", ctypes.c_int),
        peek=fn("cudaPeekAtLastError"),
        get_last=fn("cudaGetLastError"),
        device_count=fn("cudaGetDeviceCount", ctypes.POINTER(ctypes.c_int)))


def _prime(path: str) -> _PrivateRuntime:
    rt = _private_runtime(path)
    count = ctypes.c_int()
    assert rt.device_count(ctypes.byref(count)) == 0
    assert count.value <= _MISSING_ORDINAL, "the priming ordinal must not exist"
    assert rt.set_device(_MISSING_ORDINAL) != 0
    assert rt.peek() != 0, "cudaSetDevice's failure did not reach the slot"
    return rt


def _strip_clears(monkeypatch, *emitters: str) -> None:
    from tessera.compiler.emit import nvidia_cuda as N

    for name in emitters:
        original = getattr(N, name)

        def stripped(*a, _original=original, **k):
            return _original(*a, **k).replace(_CLEAR, "")

        monkeypatch.setattr(N, name, stripped)


@dataclass
class _Lane:
    """One emitted lane: how to find its library, run it, and judge a run."""

    emitters: tuple[str, ...]
    library: Callable[[], str]
    run: Callable[[], Any]           # returns the result, or raises / declines
    ok: Callable[[Any], bool]        # the run produced the real kernel's answer
    #: per-process artifact globals, as (name, factory): a fresh value per use
    reset: tuple[tuple[str, Callable[[], Any]], ...] = ()
    #: Why the negative control cannot bite for this lane, or None. A
    #: successful `cudaFuncSetAttribute` RESETS the last-error slot (measured
    #: on sm_120, CUDA 13.4 / driver 610.88), so an entry that calls it before
    #: its launch is masked from a stale error by that call. The entry still
    #: clears (the rule must not depend on undocumented reset behaviour).
    masked: str | None = None


def _rng(*shape: int, scale: float = .1) -> np.ndarray:
    return (np.random.default_rng(sum(shape)).standard_normal(shape) * scale).astype(np.float32)


def _real(ref: Callable[[], np.ndarray], atol: float) -> Callable[[Any], bool]:
    from tessera.compiler.emit.kernel_emitter import REFERENCE_EXECUTIONS

    def judge(result: Any) -> bool:
        out, tag = result
        return tag not in REFERENCE_EXECUTIONS and np.allclose(out, ref(), atol=atol, rtol=atol)
    return judge


def _timed(result: Any) -> bool:
    return result is not None and result >= 0


def _lanes() -> dict[str, _Lane]:
    from tessera.compiler import fusion_core as F
    from tessera.compiler.emit import nvidia_cuda as N
    from tessera.compiler.emit.kernel_cache import build

    fused = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32")
    fa, fb, fbias = _rng(33, 20), _rng(20, 17), _rng(17)
    attn = F.AttentionRegion(scale=16 ** -.5, causal=True)
    q, k, v = _rng(19, 16), _rng(23, 16), _rng(23, 8)
    gated = F.GatedMatmulRegion(gate_act="silu", storage_dtype="f32")
    ga, gwg, gwu = _rng(9, 12), _rng(12, 10), _rng(12, 10)
    pw = F.PointwiseGraphRegion(ops=(("add", ("x", "y"), "s"), ("relu", ("s",), "o")),
                                inputs=("x", "y"), output="o")
    px, py = _rng(4, 31), _rng(4, 31)
    mma_fused = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f16")
    mma_gated = F.GatedMatmulRegion(gate_act="silu", storage_dtype="f16")
    lowp_attn = F.AttentionRegion(scale=16 ** -.5, causal=True, storage_dtype="fp8_e4m3")
    composed = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32")
    x4 = _rng(1, 2, 16, 16)

    generic = N.NvidiaGenericCudaCandidate()
    flash = N.NvidiaFlashAttnCandidate()
    gate = N.NvidiaGatedCandidate()
    point = N.NvidiaPointwiseCandidate()
    mf = N.NvidiaMmaFusedCandidate("f16")
    ma = N.NvidiaMmaAttnCandidate("f16")
    ma8 = N.NvidiaMmaAttnCandidate("fp8_e4m3")
    mg = N.NvidiaMmaGatedCandidate("f16")
    comp = N.NvidiaMmaFusedComposedCandidate("f32")
    assert generic.applies_to(fused), "the generic lane must serve the f32 fused region"

    def lib_generic(region):
        return lambda: build(region, "nvidia", dtype="f32", dims=None).artifact

    return {
        "generic_fused": _Lane(
            ("_synthesize_fused_cuda",), lib_generic(fused),
            lambda: generic.run(fused, fa, fb, fbias),
            _real(lambda: fused.reference(fa, fb, fbias), 1e-4)),
        "generic_fused_device_ms": _Lane(
            ("_synthesize_fused_cuda",), lib_generic(fused),
            lambda: generic.measure_device_latency(fused, fa, fb, fbias, reps=3, warmup=1),
            _timed),
        "flash_attn": _Lane(
            ("_synthesize_attention_cuda",), lib_generic(attn),
            lambda: flash.run(attn, q, k, v),
            _real(lambda: attn.reference(q, k, v), 1e-4)),
        "flash_attn_device_ms": _Lane(
            ("_synthesize_attention_cuda",), lib_generic(attn),
            lambda: flash.measure_device_latency(attn, q, k, v, reps=3, warmup=1),
            _timed),
        "gated": _Lane(
            ("_synthesize_gated_cuda",), lib_generic(gated),
            lambda: gate.run(gated, ga, gwg, gwu),
            _real(lambda: gated.reference(ga, gwg, gwu), 1e-4)),
        "gated_device_ms": _Lane(
            ("_synthesize_gated_cuda",), lib_generic(gated),
            lambda: gate.measure_device_latency(gated, ga, gwg, gwu, reps=3, warmup=1),
            _timed),
        "pointwise": _Lane(
            ("_synthesize_pointwise_cuda",), lib_generic(pw),
            lambda: point.run(pw, [px, py]),
            _real(lambda: pw.reference(px, py), 1e-6)),
        "mma_fused": _Lane(
            ("_synthesize_mma_fused_cuda",),
            lambda: N._emitted_artifact(N._mma_fused_source(True, "gelu", "f16"), "f16"),
            lambda: mf.run(mma_fused, fa, fb, fbias),
            _real(lambda: mma_fused.reference(fa, fb, fbias), 5e-3)),
        "mma_fused_device_ms": _Lane(
            ("_synthesize_mma_fused_cuda",),
            lambda: N._emitted_artifact(N._mma_fused_source(True, "gelu", "f16"), "f16"),
            lambda: mf.measure_device_latency(mma_fused, fa, fb, fbias, reps=3, warmup=1),
            _timed),
        "mma_attn": _Lane(
            ("_synthesize_mma_attn_16_cuda",),
            lambda: N._emitted_artifact(N._mma_attn_source("f16"), "f16"),
            lambda: ma.run(attn, q, k, v),
            _real(lambda: attn.reference(q, k, v), 5e-3),
            masked="cudaFuncSetAttribute precedes the launch"),
        "mma_attn_fp8_device_ms": _Lane(
            ("_synthesize_mma_attn_lowp_cuda",),
            lambda: N._emitted_artifact(N._mma_attn_source("fp8_e4m3"), "fp8_e4m3"),
            lambda: ma8.measure_device_latency(lowp_attn, q, k, v, reps=3, warmup=1),
            _timed, masked="cudaFuncSetAttribute precedes the launches"),
        "mma_gated": _Lane(
            ("_synthesize_mma_gated_cuda",),
            lambda: N._emitted_artifact(N._mma_gated_source("f16", "silu"), "f16"),
            lambda: mg.run(mma_gated, ga, gwg, gwu),
            _real(lambda: mma_gated.reference(ga, gwg, gwu), 2e-2)),
        "mma_gated_device_ms": _Lane(
            ("_synthesize_mma_gated_cuda",),
            lambda: N._emitted_artifact(N._mma_gated_source("f16", "silu"), "f16"),
            lambda: mg.measure_device_latency(mma_gated, ga, gwg, gwu, reps=3, warmup=1),
            _timed),
        "resident_stages": _Lane(
            ("_synthesize_resident_ops_cuda",),
            lambda: N._emitted_artifact(N._resident_ops_source(), "f32"),
            lambda: comp.run(composed, fa, fb, fbias),
            _real(lambda: composed.reference(fa, fb, fbias), 2e-2)),
        "binary": _Lane(
            ("_synthesize_binary_cuda",), lambda: N._binary_artifact,
            lambda: N.run_binary_arithmetic(px, py, 5),          # kind 5: a + b
            lambda out: np.allclose(out, px + py, atol=1e-6),
            reset=(("_binary_artifact", lambda: None),)),
        "solver_ift": _Lane(
            ("_synthesize_solver_ift_cuda",), lambda: N._solver_ift_artifact,
            lambda: N.run_solver_ift_f32(np.abs(px) + 1, np.sqrt(np.abs(px) + 1), px),
            lambda out: all(np.all(np.isfinite(o)) for o in out),
            reset=(("_solver_ift_artifact", lambda: None),)),
        "solver_unary": _Lane(
            ("_synthesize_solver_children_cuda",), lambda: N._solver_child_artifact,
            lambda: N.run_solver_unary(px, 4),                   # kind 4: tanh
            lambda out: np.allclose(out, np.tanh(px), atol=1e-6),
            reset=(("_solver_child_artifact", lambda: None),)),
        "flash_multiwarp_timed": _Lane(
            ("_synthesize_flash_fwd_multiwarp_cuda",),
            lambda: N._flash_fwd_schedule_artifact[4],
            lambda: N.measure_flash_attention_forward_schedule_device(
                x4, x4, x4, scale=.25, warps_per_cta=4, warmup=1, reps=3),
            lambda ms: ms >= 0,
            reset=(("_flash_fwd_schedule_artifact", dict),)),
    }


def _attempt(lane: _Lane) -> tuple[bool, Any]:
    try:
        result = lane.run()
    except Exception as error:          # a lane that raises on rc != 1
        return False, error
    return lane.ok(result), result


_LANE_NAMES = (
    "generic_fused", "generic_fused_device_ms", "flash_attn", "flash_attn_device_ms",
    "gated", "gated_device_ms", "pointwise", "mma_fused", "mma_fused_device_ms",
    "mma_attn", "mma_attn_fp8_device_ms", "mma_gated", "mma_gated_device_ms",
    "resident_stages", "binary", "solver_ift", "solver_unary", "flash_multiwarp_timed")


@pytest.mark.parametrize("name", _LANE_NAMES)
def test_primed_stale_error_does_not_fail_the_lane(monkeypatch, name):
    _require_sm120()
    from tessera.compiler.emit import nvidia_cuda as N

    lane = _lanes()[name]

    # 1. Negative control: without the entry clear the prime must bite.
    with monkeypatch.context() as patched:
        for attr, fresh in lane.reset:
            patched.setattr(N, attr, fresh())
        _strip_clears(patched, *lane.emitters)
        ok, result = _attempt(lane)          # compile + load the stripped lane
        assert ok, f"{name}: the stripped lane must work on a clean slot: {result!r}"
        stripped_lib = lane.library()
        runtime = _prime(stripped_lib)
        ok, result = _attempt(lane)
        runtime.get_last()                   # leave nothing behind either way
        assert lane.masked or not ok, (
            f"{name}: with the clear stripped a primed slot did not fail the lane "
            f"({result!r}) -- the prime does not reach the slot it reads, so the "
            "positive half below would prove nothing")

    # 2. The shipped emitter under the same prime.
    for attr, fresh in lane.reset:
        monkeypatch.setattr(N, attr, fresh())
    ok, result = _attempt(lane)
    assert ok, f"{name}: the shipped lane fails on a clean slot: {result!r}"
    library = lane.library()
    assert library != stripped_lib, "the stripped source must compile separately"
    runtime = _prime(library)
    ok, result = _attempt(lane)
    assert ok, f"{name}: a stale error failed the shipped lane: {result!r}"
    assert runtime.peek() == 0, f"{name}: the lane left an error in its slot"


def test_a_refused_allocation_does_not_fail_the_next_launch():
    """The realistic trigger, through the lane's own ABI: a workload whose
    device allocation is refused leaves cudaErrorMemoryAllocation in the
    library's slot (the entry returns early without reading it); the next,
    ordinary launch must still succeed."""
    _require_sm120()
    from tessera.compiler import fusion_core as F
    from tessera.compiler.emit import nvidia_cuda as N

    region = F.FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f16")
    a, b, bias = _rng(33, 20), _rng(20, 17), _rng(17)
    cand = N.NvidiaMmaFusedCandidate("f16")
    out, tag = cand.run(region, a, b, bias)
    assert tag == "nvidia_cuda"
    fn = N._mma_fused_fn(True, "gelu", "f16")
    runtime = _private_runtime(N._emitted_artifact(N._mma_fused_source(True, "gelu", "f16"), "f16"))
    huge = 1 << 20                        # M*K*2 bytes = 2 TiB: refused, never touched
    assert fn(None, None, None, None, huge, 8, huge) == 3
    assert runtime.peek() != 0, "the refused allocation must leave the slot set"
    out, tag = cand.run(region, a, b, bias)
    assert tag == "nvidia_cuda", "a refused allocation failed the next correct launch"
    assert np.allclose(out, region.reference(a, b, bias), atol=5e-3, rtol=5e-3)
