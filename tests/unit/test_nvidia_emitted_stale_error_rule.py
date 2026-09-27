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

Every ``_synthesize_*`` emitter must be listed in the shared table, so a new
one cannot skip the gate. Host-independent (renders source text only); the
sm_120 behaviour is proven by ``tests/device/nvidia/test_emitted_stale_cuda_error.py``.
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


@pytest.mark.parametrize("function", S.INLINE_SOURCE_FUNCTIONS)
def test_inline_sources_never_read_the_slot(function):
    text = S.inline_source_text(function)
    assert 'extern "C"' in text, f"{function} no longer holds an inline source"
    assert "cudaGetLastError" not in text and "cudaPeekAtLastError" not in text, (
        f"{function}'s inline CUDA reads the last-error slot: move it to an "
        "emitter the gate renders, and clear on entry")


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
