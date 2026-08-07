from __future__ import annotations

import copy

import pytest

from tessera.compiler.profiler_x86_evidence import (
    X86ProfilerPacketError,
    build_x86_profiler_packet,
    validate_x86_profiler_packet,
)


def _benchmark() -> dict[str, object]:
    return {
        "schema": "tessera.compiler.e2e_real4.x86_matmul.v1",
        "architecture": "zen5-avx512",
        "rows": [
            {"shape_class": "aligned", "correctness": {"passed": True}},
            {"shape_class": "ragged", "correctness": {"passed": True}},
        ],
        "verdict": "promote",
    }


def _timing() -> dict[str, object]:
    return {
        name: True
        for name in (
            "avx512_visible",
            "monotonic_raw_valid",
            "rdtscp_valid",
            "invariant_tsc",
            "affinity_stable",
            "clock_agreement_valid",
            "perf_event_open",
            "perf_sample_valid",
        )
    }


def _cpu() -> dict[str, object]:
    return {
        "vendor_id": "AuthenticAMD",
        "cpu_family": 26,
        "model_name": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        "flags": ["avx512f"],
    }


def _environment(*, wsl: bool) -> dict[str, object]:
    return {
        "wsl": wsl,
        "virtualized": wsl,
        "source_commit": "a" * 40,
        "worktree_dirty": False,
    }


def test_wsl_zen5_packet_is_retain_not_promotion() -> None:
    packet = build_x86_profiler_packet(
        benchmark=_benchmark(),
        timing_status=_timing(),
        cpu=_cpu(),
        environment=_environment(wsl=True),
        sampling=None,
    )
    assert packet["verdict"] == "retain"
    assert packet["eligible_for_promotion"] is False
    assert "WSL_CLOCK_DOMAIN" in packet["ineligibility_reasons"]
    assert "SYMBOL_SAMPLING_MISSING" in packet["ineligibility_reasons"]
    validate_x86_profiler_packet(packet)


def test_packet_digest_and_shape_coverage_fail_closed() -> None:
    packet = build_x86_profiler_packet(
        benchmark=_benchmark(),
        timing_status=_timing(),
        cpu=_cpu(),
        environment=_environment(wsl=False),
        sampling=None,
    )
    tampered = copy.deepcopy(packet)
    tampered["benchmark"]["verdict"] = "reject"
    with pytest.raises(X86ProfilerPacketError, match="digest"):
        validate_x86_profiler_packet(tampered)

    benchmark = _benchmark()
    benchmark["rows"] = [benchmark["rows"][0]]
    with pytest.raises(X86ProfilerPacketError, match="aligned and ragged"):
        build_x86_profiler_packet(
            benchmark=benchmark,
            timing_status=_timing(),
            cpu=_cpu(),
            environment=_environment(wsl=False),
            sampling=None,
        )
