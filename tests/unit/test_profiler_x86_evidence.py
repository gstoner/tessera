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


# --- tsc_witness route (sync WSL-TIMING-ADMISSION-2026-09-26, x86 follow-up) ---

_HZ = 3_000_000_000.0


@pytest.fixture(autouse=True)
def _invariant_tsc_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The witness records /proc/cpuinfo's invariant-TSC flags; pin them so
    these synthetic samples mean the same thing on every host."""
    from tessera.compiler import profiler_x86_clock
    monkeypatch.setattr(profiler_x86_clock, "invariant_tsc", lambda: True)


def _witness(image: str, *, raw_ns: int = 50_000_000, tsc_scale: float = 1.0,
             digest_image: str | None = None, env: str = "wsl2") -> dict[str, object]:
    from tessera.compiler.profiler_x86_clock import witness_sample

    calibration = {"frequency_hz": _HZ, "spread": 1e-5, "rates_hz": [_HZ] * 5,
                   "windows": [{"raw_start_ns": 0, "raw_end_ns": 200_000_000}]}
    window = {"raw_start_ns": 1_000_000_000, "raw_end_ns": 1_000_000_000 + raw_ns,
              "tsc_start": 10, "tsc_end": 10 + int(raw_ns * _HZ / 1e9 * tsc_scale),
              "logical_cpu_start": 3, "logical_cpu_end": 3, "host_wall_ns": raw_ns + 1000}
    return witness_sample(calibration, window, {"image": digest_image or image},
                          execution_environment=env)


def _witnessed_benchmark(**overrides: object) -> dict[str, object]:
    benchmark = _benchmark()
    for index, row in enumerate(benchmark["rows"]):  # type: ignore[union-attr]
        image = f"{index:064x}"
        row["compile"] = {"digests": {"image": image}}
        row["timing_witness"] = _witness(image, **overrides)  # type: ignore[arg-type]
    return benchmark


def _packet(benchmark: dict[str, object], **timing: bool) -> dict[str, object]:
    return build_x86_profiler_packet(
        benchmark=benchmark, timing_status={**_timing(), **timing}, cpu=_cpu(),
        environment=_environment(wsl=True), sampling=None)


def test_every_row_witnessed_takes_the_tsc_route_and_demotes_environment_tags() -> None:
    packet = _packet(_witnessed_benchmark(), rdtscp_valid=False, clock_agreement_valid=False)
    assert packet["admission_route"] == "tsc_witness"
    assert packet["ineligibility_reasons"] == []
    assert packet["eligible_for_promotion"] is True
    gap_tags = {gap.partition(":")[0] for gap in packet["diagnostic_gaps"]}
    assert {"WSL_CLOCK_DOMAIN", "SYMBOL_SAMPLING_MISSING", "TIMING_PROOF_INCOMPLETE"} <= gap_tags
    validate_x86_profiler_packet(packet)


def test_avx512_visibility_still_blocks_on_the_tsc_route() -> None:
    packet = _packet(_witnessed_benchmark(), avx512_visible=False)
    assert packet["admission_route"] == "tsc_witness"
    assert packet["eligible_for_promotion"] is False
    assert any("avx512_visible" in reason for reason in packet["ineligibility_reasons"])
    validate_x86_profiler_packet(packet)


@pytest.mark.parametrize("damage", ["missing_row", "disagreeing_tsc", "other_image"])
def test_one_bad_row_keeps_the_profiler_route(damage: str) -> None:
    benchmark = _witnessed_benchmark()
    row = benchmark["rows"][1]  # type: ignore[index]
    if damage == "missing_row":
        del row["timing_witness"]
    elif damage == "disagreeing_tsc":
        row["timing_witness"] = _witness(row["compile"]["digests"]["image"], tsc_scale=1.10)
    else:
        row["timing_witness"] = _witness(row["compile"]["digests"]["image"], digest_image="f" * 64)
    packet = _packet(benchmark)
    assert packet["admission_route"] == "profiler_correlated"
    assert packet["eligible_for_promotion"] is False
    assert "WSL_CLOCK_DOMAIN" in packet["ineligibility_reasons"]
    validate_x86_profiler_packet(packet)


def test_validator_rederives_route_and_reason_split() -> None:
    packet = _packet(_witnessed_benchmark())
    forged = copy.deepcopy(_packet(_benchmark()))
    forged["admission_route"] = "tsc_witness"
    with pytest.raises(X86ProfilerPacketError, match="admission route"):
        validate_x86_profiler_packet(forged)
    for field_name, value in (("diagnostic_gaps", []), ("ineligibility_reasons", ["VIRTUALIZED_HOST"])):
        moved = copy.deepcopy(packet)
        moved[field_name] = value
        with pytest.raises(X86ProfilerPacketError, match="reasons/gaps|has blockers"):
            validate_x86_profiler_packet(moved)
