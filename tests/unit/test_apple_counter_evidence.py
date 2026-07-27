"""APPLE-COUNTER-1 — Metal-native evidence for the shared autotune schema.

The cross-backend sync `NVIDIA-TEST5-2026-07-16` asked Apple to populate the
same logical evidence fields from Metal counters. The risk it creates is subtle:
a field that CUDA fills from `ptxas` and Metal cannot answer at all must not
quietly become `0`, because a selector reading `spill_count = 0` would conclude
"no spills" from what is really "no query".

These tests pin the four-state distinction — measured / not measured this run /
unsupported by this device / no public API — and keep the capability bits in
step with the runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.apple_counter_evidence import (
    CAP_DISPATCH_BOUNDARY_SAMPLING,
    CAP_METAL4_TIMESTAMP_HEAP,
    CAP_PIPELINE_LIMITS,
    CAP_TIMESTAMP_COUNTER_SET,
    NO_PUBLIC_METAL_QUERY,
    AppleEvidenceState,
    AppleProfilingCapabilities,
    apple_autotune_evidence,
    evidence_from_dispatch_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
)

FULL = AppleProfilingCapabilities(
    CAP_PIPELINE_LIMITS | CAP_TIMESTAMP_COUNTER_SET
    | CAP_DISPATCH_BOUNDARY_SAMPLING | CAP_METAL4_TIMESTAMP_HEAP)
# This M1 Max reports pipeline limits and the timestamp counter *set*, but not
# dispatch-boundary sampling — the exact shape the TILE-1 rungs recorded.
NO_SAMPLING = AppleProfilingCapabilities(
    CAP_PIPELINE_LIMITS | CAP_TIMESTAMP_COUNTER_SET)


def test_capability_bits_match_the_runtime_definition():
    """The Python bit positions mirror `tessera_apple_gpu_profiling_capabilities`."""
    source = RUNTIME_SOURCE.read_text(encoding="utf-8", errors="replace")
    block = source[source.index("APPLE-GEMM-1 profiling capability matrix"):]
    block = block[:block.index("extern \"C\" int32_t tessera_apple_gpu_profiling_capabilities")]
    documented = {int(bit): name.strip()
                  for bit, name in re.findall(r"//\s+(\d):\s*(.+)", block)}
    assert documented, "runtime no longer documents its capability bits"
    assert documented[0].startswith("compiled-pipeline limits")
    assert documented[1].startswith("timestamp counter set")
    assert documented[4].startswith("dispatch-boundary counter sampling")
    assert documented[6].startswith("Metal 4 timestamp heap")
    # Positions, not just names.
    assert CAP_PIPELINE_LIMITS == 1 << 0
    assert CAP_TIMESTAMP_COUNTER_SET == 1 << 1
    assert CAP_DISPATCH_BOUNDARY_SAMPLING == 1 << 4
    assert CAP_METAL4_TIMESTAMP_HEAP == 1 << 6


def test_fields_metal_cannot_answer_are_absent_with_a_reason():
    """`no_public_api` never degrades to a measured zero."""
    evidence = apple_autotune_evidence(capabilities=FULL, device_time_ns=21_000)
    for field in NO_PUBLIC_METAL_QUERY:
        assert evidence[field]["value"] is None
        assert evidence[field]["state"] == AppleEvidenceState.NO_PUBLIC_API.value
    # The permanent gaps are exactly the ptxas-sourced ones.
    assert set(NO_PUBLIC_METAL_QUERY) == {
        "register_count", "scratch_bytes", "spill_count", "achieved_occupancy"}


def test_counter_delta_is_a_device_gap_not_a_missing_measurement():
    """Absent sampling support is recorded as a device limitation."""
    evidence = apple_autotune_evidence(capabilities=NO_SAMPLING, device_time_ns=21_000)
    delta = evidence["counter_timestamp_delta"]
    assert delta["value"] is None
    assert delta["state"] == AppleEvidenceState.UNSUPPORTED_BY_DEVICE.value
    # A device that *can* sample but did not this run is a different state, so
    # a later better-instrumented run is distinguishable from a hardware limit.
    supported = apple_autotune_evidence(capabilities=FULL, device_time_ns=21_000)
    assert (supported["counter_timestamp_delta"]["state"]
            == AppleEvidenceState.NOT_MEASURED.value)
    # Kernel time still comes from the command buffer, which needs no counters.
    assert evidence["device_time_ns"]["value"] == 21_000
    assert evidence["device_time_ns"]["state"] == AppleEvidenceState.MEASURED.value


def test_claiming_a_counter_delta_without_sampling_is_rejected():
    """A corpus row cannot assert evidence the device cannot produce."""
    with pytest.raises(ValueError, match="APPLE_COUNTER_EVIDENCE_UNSUPPORTED"):
        apple_autotune_evidence(
            capabilities=NO_SAMPLING, counter_timestamp_delta=1234)


def test_claiming_pipeline_limits_without_support_is_rejected():
    bare = AppleProfilingCapabilities(0)
    with pytest.raises(ValueError, match="APPLE_COUNTER_EVIDENCE_UNSUPPORTED"):
        apple_autotune_evidence(capabilities=bare, execution_width=32)


def test_measured_fields_carry_their_values_and_state():
    evidence = apple_autotune_evidence(
        capabilities=FULL,
        device_time_ns=23_100,
        counter_timestamp_delta=19_800,
        threadgroup_memory_bytes=4352,
        execution_width=32,
        max_threads_per_threadgroup=1024,
        pipeline_cache_size=3,
        cold_compile=True,
        source_sha256="sha256:abc",
    )
    measured = AppleEvidenceState.MEASURED.value
    assert evidence["counter_timestamp_delta"] == {"value": 19_800, "state": measured}
    assert evidence["static_threadgroup_memory_bytes"] == {"value": 4352, "state": measured}
    assert evidence["cold_compile"] == {"value": True, "state": measured}
    assert evidence["source_fingerprint"] == {"value": "sha256:abc", "state": measured}


def test_dispatch_record_adapts_without_teaching_it_about_nvidia():
    """The Tile lane's own record maps onto the shared schema unchanged."""

    class _Record:
        device_time_ns = 22_400
        counter_timestamp_delta = None
        msl_pipeline_cache_size = 2
        source_sha256 = "sha256:def"
        resources = {"total_threadgroup_bytes": 4352, "simdgroup_lanes": 32}

    evidence = evidence_from_dispatch_record(
        _Record(), NO_SAMPLING, cold_compile=False)
    assert evidence["device_time_ns"]["value"] == 22_400
    assert evidence["static_threadgroup_memory_bytes"]["value"] == 4352
    assert evidence["execution_width"]["value"] == 32
    assert evidence["pipeline_cache_size"]["value"] == 2
    assert evidence["cold_compile"]["value"] is False
    assert (evidence["counter_timestamp_delta"]["state"]
            == AppleEvidenceState.UNSUPPORTED_BY_DEVICE.value)
    assert evidence["spill_count"]["state"] == AppleEvidenceState.NO_PUBLIC_API.value


@pytest.mark.hardware_apple_gpu
def test_live_device_reports_a_capability_matrix():
    """The probe returns a real matrix on Metal, and a probed zero is not None."""
    from tessera.compiler.apple_counter_evidence import (
        probe_apple_profiling_capabilities,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    capabilities = probe_apple_profiling_capabilities()
    assert capabilities is not None, "a Metal host must report a capability matrix"
    # Every Apple GPU can report its compiled-pipeline limits.
    assert capabilities.pipeline_limits
    # Whatever the sampling answer is, the evidence must be consistent with it.
    evidence = apple_autotune_evidence(
        capabilities=capabilities, device_time_ns=1)
    # No delta was supplied, so the state must say *why* it is absent: a device
    # that cannot sample reports the hardware limit, one that can reports that
    # this particular run did not capture it.
    expected = (AppleEvidenceState.NOT_MEASURED.value
                if capabilities.counter_deltas_available
                else AppleEvidenceState.UNSUPPORTED_BY_DEVICE.value)
    assert evidence["counter_timestamp_delta"]["state"] == expected
