"""The stale-error rule holds in every CUDA source ``nvidia_cuda`` emits.

Sync keys ``SPECTRAL-STALE-HIP-ERROR-2026-09-27`` (the rule, hand-written
hooks) and ``SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`` (the emitted templates).
The CUDA last-error slot is per thread and per runtime instance and only
``cudaGetLastError()`` resets it, so an entry that reads it after its own
launch reports any older unread error as its own failure. The rule:

* in a source that reads the slot, every exported entry that does device work
  clears it exactly once, as its first statement, and nowhere else -- never
  between a launch and its check;
* entries that make no call at all (pointer getters) do not clear;
* a source that never reads the slot carries no clear (nothing there could
  be misled, and an inert call is churn in every identity);
* the arbiter-raced lanes check each launch group through the slot, after the
  launch, so a launch that never ran cannot be timed or served as a kernel.

Extended by ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (``NVIDIA-EMITTED-UNCHECKED-LAUNCH``):
the rule now covers every emitted source, not only the raced lanes. A launch
judged by ``cudaDeviceSynchronize`` alone ("sync-only") is no longer an
accepted class, because on sm_120 the sync returns success over a launch that
never ran. Every runtime launch is read through the slot (a timer's warm-up
group before its timing boundary), and every allocation / copy / memset /
event / driver-launch status is consumed rather than discarded.

Every ``_synthesize_*`` emitter must be listed in the shared table, and no
CUDA kernel source may live outside an emitter, so nothing can skip the gate.
Host-independent (renders source text only); the sm_120 behaviour is proven by
``tests/device/nvidia/test_emitted_stale_cuda_error.py`` and
``tests/device/nvidia/test_emitted_unchecked_launch.py``.
"""
from __future__ import annotations

import re

import pytest

from tests._support import nvidia_emitted_sources as S

#: Exported entries of the arbiter-raced lanes, and how many launch groups each
#: must check through the slot (a host entry: its launch; a device timer: its
#: warm-up group and its timed group).
_RACED = {
    "tessera_nvidia_fused": 1, "tessera_nvidia_fused_device_ms": 2,
    "tessera_nvidia_attn": 1, "tessera_nvidia_attn_device_ms": 2,
    "tessera_nvidia_gated": 1, "tessera_nvidia_gated_device_ms": 2,
    "tessera_nvidia_pointwise": 1,
    "tessera_nvidia_mma_fused": 1, "tessera_nvidia_mma_fused_device_ms": 2,
    "tessera_nvidia_mma_attn": 1, "tessera_nvidia_mma_attn_device_ms": 2,
    "tessera_nvidia_mma_gated": 1, "tessera_nvidia_mma_gated_device_ms": 2,
}


def _sources():
    return S.rendered_sources()


def test_every_emitter_is_in_the_table():
    listed = set(S._emitters())
    defined = S.emitter_names_in_module()
    assert defined - listed == set(), (
        "emitters not covered by the stale-error gate -- add them to "
        f"tests/_support/nvidia_emitted_sources.py: {sorted(defined - listed)}")
    assert listed - defined == set(), f"stale table entries: {sorted(listed - defined)}"


def test_no_cuda_source_lives_outside_an_emitter():
    """A kernel source built inline in a run function is invisible to every
    rule below. The two that existed (`run_relu_bias_f32`,
    `run_fused_epilogue_f32`) were sync-only; they now come from emitters."""
    assert S.INLINE_SOURCE_FUNCTIONS == ()
    offenders = S.functions_holding_kernel_source()
    assert offenders == [], (
        f"CUDA kernel source outside a `_synthesize_*` emitter: {offenders}; "
        "move it to an emitter and list it in tests/_support/nvidia_emitted_sources.py")


def test_no_entry_is_sync_only():
    """Every entry that makes a call is slot-checked or status-checked; none
    judges its launch by `cudaDeviceSynchronize` alone."""
    rows = S.classify()
    bad = [f"{r['label']}:{r['entry']}" for r in rows if r["kind"] == "sync-only"]
    assert not bad, "sync-only entries (launch judged by the sync alone):\n" + "\n".join(bad)
    assert {r["kind"] for r in rows} <= {"metadata", "slot-checked", "status-checked"}


def test_every_runtime_launch_is_checked_through_the_slot():
    bad = []
    for emitter, label, source in _sources():
        for fn in S.host_functions(source):
            for problem in S.launch_check_violations(fn):
                bad.append(f"{label}:{fn.name}: {problem}")
    assert not bad, "\n".join(bad)


def test_every_status_is_consumed():
    """Allocation, copy, memset, event and driver-launch statuses are checked,
    never discarded as a bare statement (a failed H2D copy leaves the kernel
    reading garbage and the entry reporting success)."""
    bad = []
    for emitter, label, source in _sources():
        for fn in S.host_functions(source):
            for call in S.unchecked_status_calls(fn):
                bad.append(f"{label}:{fn.name}: {call} status discarded")
    assert not bad, "\n".join(bad)


def test_every_entry_in_a_slot_reading_source_clears_first_exactly_once():
    bad = []
    for emitter, label, source in _sources():
        if not S.source_reads_slot(source):
            continue
        for entry in S.exported_entries(source):
            if not entry.makes_calls():
                if entry.clears():
                    bad.append(f"{label}:{entry.name} clears but makes no call")
                continue
            if not entry.first_statement_is_clear():
                bad.append(f"{label}:{entry.name} does not clear as its first statement")
            if entry.clears() != 1:
                bad.append(f"{label}:{entry.name} clears {entry.clears()} times")
    assert not bad, "\n".join(bad)


def test_no_clear_outside_an_entry_prologue():
    """A clear anywhere else -- between a launch and its check, in a helper, or
    in a source that never reads the slot -- is a defect or churn."""
    bad = []
    for emitter, label, source in _sources():
        prologues = sum(e.first_statement_is_clear() for e in S.exported_entries(source))
        total = source.count(S.CLEAR)
        if total != prologues:
            bad.append(f"{label}: {total} clears, {prologues} in entry prologues")
        if total and not S.source_reads_slot(source):
            bad.append(f"{label}: clears but never reads the slot")
    assert not bad, "\n".join(bad)


def test_raced_lanes_check_every_launch_group_after_launching():
    seen: set[str] = set()
    bad = []
    for emitter, label, source in _sources():
        for entry in S.exported_entries(source):
            groups = _RACED.get(entry.name)
            if groups is None:
                continue
            seen.add(entry.name)
            body = entry.body
            if entry.reads() < groups:
                bad.append(f"{label}:{entry.name} checks {entry.reads()} of {groups} "
                           "launch groups")
            last_launch = body.rfind("<<<")
            reads_after = [m.start() for m in re.finditer(r"cudaGetLastError\(\)", body)
                           if m.start() > last_launch]
            if not reads_after:
                bad.append(f"{label}:{entry.name} never reads the slot after its last launch")
            if groups == 2:
                # A timer checks its warm-up group before it starts timing:
                # a read strictly between its first and last launch.
                first_launch = body.find("<<<")
                between = [m.start() for m in re.finditer(r"cudaGetLastError\(\)", body)
                           if first_launch < m.start() < last_launch]
                if not between:
                    bad.append(f"{label}:{entry.name} does not check its warm-up "
                               "launches before the timed group")
    assert not bad, "\n".join(bad)
    assert seen == set(_RACED), f"raced entries not rendered: {sorted(set(_RACED) - seen)}"


def test_scalar_lanes_have_a_device_timer_entry():
    """AUTOTUNE sm_120 follow-up: the three scalar lanes that raced device rows
    unmeasured now carry a `_device_ms` entry beside their host entry, in the
    same source (so one artifact serves `run` and the timer)."""
    by_label = {label: source for _, label, source in _sources()}
    for label, entry in (("fused(bias,gelu)", "tessera_nvidia_fused"),
                         ("attention", "tessera_nvidia_attn"),
                         ("gated(silu)", "tessera_nvidia_gated")):
        names = {e.name for e in S.exported_entries(by_label[label])}
        assert {entry, f"{entry}_device_ms"} <= names, (label, sorted(names))


def test_the_parser_sees_what_it_is_asked_about():
    """Guard the gate itself: a clear-less reading entry and a misplaced clear
    must both be reported (a gate that parses nothing passes everything)."""
    src = ('extern "C" int a(int*p){k<<<1,1>>>(p);return cudaGetLastError()?3:1;}\n'
           'extern "C" int b(int*p){k<<<1,1>>>(p);(void)cudaGetLastError();'
           'return cudaGetLastError()?3:1;}\n'
           'extern "C" void* c(void*v){return v;}\n')
    entries = {e.name: e for e in S.exported_entries(src)}
    assert set(entries) == {"a", "b", "c"}
    assert S.source_reads_slot(src)
    assert not entries["a"].first_statement_is_clear()
    assert not entries["b"].first_statement_is_clear() and entries["b"].clears() == 1
    assert not entries["c"].makes_calls()
    assert entries["a"].reads() == 1 and entries["b"].reads() == 1


def test_launch_and_status_parsers_see_what_they_are_asked_about():
    """Guard the launch/status half of the gate the same way: each defect it
    exists to catch must be reported, and the fixed form must pass."""
    src = (
        '#include <cuda_runtime.h>\n'
        '__global__ void k(int*p){if(p)*p=1;}\n'
        '// a { brace and cudaMemcpy( in a comment are not code\n'
        'static int helper(int*p){k<<<1,1>>>(p);return cudaGetLastError()?3:1;}\n'
        'extern "C" int sync_only(int*h){int*p=0;if(cudaMalloc(&p,4))return 2;'
        'cudaMemcpy(p,h,4,cudaMemcpyHostToDevice);k<<<1,1>>>(p);'
        'return cudaDeviceSynchronize()==cudaSuccess?1:3;}\n'
        'extern "C" int fixed(int*h){(void)cudaGetLastError();int*p=0;'
        'if(cudaMalloc(&p,4)||cudaMemcpy(p,h,4,cudaMemcpyHostToDevice))return 2;'
        'k<<<1,1>>>(p);return cudaGetLastError()==cudaSuccess&&'
        'cudaDeviceSynchronize()==cudaSuccess?1:3;}\n'
        'extern "C" int timer(int*p,float*ms){(void)cudaGetLastError();cudaEvent_t a,b;'
        'k<<<1,1>>>(p);if(cudaEventCreate(&a)||cudaEventCreate(&b)||cudaEventRecord(a))return 3;'
        'k<<<1,1>>>(p);if(cudaEventRecord(b)||cudaEventSynchronize(b)||cudaGetLastError()||'
        'cudaEventElapsedTime(ms,a,b))return 3;return 1;}\n'
        'extern "C" void dl(int*p){cudaDeviceSynchronize();cudaFree(p);}\n')
    fns = {f.name: f for f in S.host_functions(src)}
    assert set(fns) == {"helper", "sync_only", "fixed", "timer", "dl"}, set(fns)
    assert not fns["helper"].exported and fns["fixed"].exported
    assert S.launch_check_violations(fns["helper"]) == []
    assert S.launch_check_violations(fns["sync_only"]) == [
        "its last launch is never read through the slot"]
    assert S.unchecked_status_calls(fns["sync_only"]) == ["cudaMemcpy"]
    assert S.launch_check_violations(fns["fixed"]) == []
    assert S.unchecked_status_calls(fns["fixed"]) == []
    # The timer's warm-up launch reaches `cudaEventRecord(a)` unread.
    assert S.launch_check_violations(fns["timer"]) == [
        "a launch group reaches a timing boundary unread"]
    assert fns["dl"].returns_void and S.unchecked_status_calls(fns["dl"]) == []
    reset = S.HostFunction(name="r", signature='extern "C" int r()', body=(
        "k<<<1,1>>>(0);cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,0);"
        "return cudaGetLastError()?3:1;"))
    assert S.launch_check_violations(reset) == [
        "cudaFuncSetAttribute sits between a launch and its read"]
