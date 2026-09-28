"""Launch integrity for every HIP source ``emit/rocm_hip.py`` emits.

Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (the ROCm half of
``NVIDIA-EMITTED-UNCHECKED-LAUNCH``). The ROCm emitter already cleared the
sticky per-thread HIP slot in every entry that read it
(``SPECTRAL-STALE-HIP-ERROR-2026-09-27``), but three entries launched without
reading it at all and judged their launch by ``hipDeviceSynchronize`` alone: the
ReplaySSM ``su`` step, the paged-KV gather and the direct paged attention -- the
two routes the gfx1151 paged-KV warm start serves. Their event-timer calls
(``hipEventRecord`` / ``hipEventSynchronize``) were unchecked too.

Rules, the same as the CUDA gate's (``test_nvidia_emitted_stale_error_rule.py``):

* every runtime launch (``hipLaunchKernelGGL`` or ``<<<``) is read through the
  slot after it, and a timer's warm-up group before its timing boundary;
* every allocation / copy / memset / event status is consumed;
* every host function that launches clears the slot before its first launch
  (after host-only argument validation or its own checked copies -- the ROCm
  emitter's convention; nothing the clear drops was unread, by the rule above).

Host-independent: renders source text only. Device behaviour on gfx1151 is the
ROCm paged-KV re-record and the ROCm unit lanes that run these entries.
"""
from __future__ import annotations

from tests._support import nvidia_emitted_sources as S


def _sources():
    return S.rocm_rendered_sources()


def test_every_hip_emitter_is_rendered():
    names = {emitter for emitter, _, _ in _sources()}
    assert {"_synthesize_paged_kv_read_hip", "_synthesize_paged_attention_direct_hip",
            "_synthesize_ssm_replay_device_hip", "_synthesize_fused_hip"} <= names


def test_every_hip_launch_is_checked_through_the_slot():
    bad = []
    for _, label, source in _sources():
        for fn in S.host_functions(source):
            for problem in S.launch_check_violations(fn, S.HIP):
                bad.append(f"{label}:{fn.name}: {problem}")
    assert not bad, "\n".join(bad)


def test_every_hip_status_is_consumed():
    bad = []
    for _, label, source in _sources():
        for fn in S.host_functions(source):
            for call in S.unchecked_status_calls(fn, S.HIP):
                bad.append(f"{label}:{fn.name}: {call} status discarded")
    assert not bad, "\n".join(bad)


def test_every_launching_hip_function_clears_before_its_first_launch():
    bad = []
    for _, label, source in _sources():
        for fn in S.host_functions(source):
            if S.HIP.launch.search(fn.body) and not S.clears_before_first_launch(fn, S.HIP):
                bad.append(f"{label}:{fn.name}")
    assert not bad, "launching functions that do not clear the HIP slot first:\n" + "\n".join(bad)


def test_the_hip_rules_see_what_they_are_asked_about():
    """A launch judged by the sync alone, an unchecked event record and a
    missing clear must each be reported; the fixed form must pass."""
    src = (
        '#include <hip/hip_runtime.h>\n'
        '__global__ void k(int*p){if(p)*p=1;}\n'
        'extern "C" int bad(int*p,float*ms){hipEvent_t a,b;'
        'hipLaunchKernelGGL(k,dim3(1),dim3(1),0,0,p);if(hipDeviceSynchronize()!=hipSuccess)return 3;'
        'hipEventRecord(a,0);k<<<1,1>>>(p);hipEventRecord(b,0);'
        'return hipEventElapsedTime(ms,a,b)==hipSuccess?1:3;}\n'
        'extern "C" int good(int*p){if(!p)return 2;(void)hipGetLastError();'
        'hipLaunchKernelGGL(k,dim3(1),dim3(1),0,0,p);'
        'return hipGetLastError()==hipSuccess&&hipDeviceSynchronize()==hipSuccess?1:3;}\n')
    fns = {f.name: f for f in S.host_functions(src)}
    assert S.launch_check_violations(fns["bad"], S.HIP) == [
        "a launch group reaches a timing boundary unread",
        "its last launch is never read through the slot"]
    assert S.unchecked_status_calls(fns["bad"], S.HIP) == ["hipEventRecord", "hipEventRecord"]
    assert not S.clears_before_first_launch(fns["bad"], S.HIP)
    assert S.launch_check_violations(fns["good"], S.HIP) == []
    assert S.unchecked_status_calls(fns["good"], S.HIP) == []
    assert S.clears_before_first_launch(fns["good"], S.HIP)
