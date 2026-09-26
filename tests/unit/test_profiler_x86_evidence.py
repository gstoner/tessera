from __future__ import annotations

import copy
from typing import Any

import pytest

from tessera.compiler.profiler_x86_evidence import (
    X86ProfilerPacketError,
    build_x86_profiler_packet,
    digest_json,
    validate_x86_profiler_packet,
)

_HZ = 3_000_000_000.0
_CPU = 3
_SAMPLE_MS = 5.0
_TRIALS = 3


@pytest.fixture(autouse=True)
def _invariant_tsc_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The witness records /proc/cpuinfo's invariant-TSC flags; pin them so
    these synthetic samples mean the same thing on every host."""
    from tessera.compiler import profiler_x86_clock
    monkeypatch.setattr(profiler_x86_clock, "invariant_tsc", lambda: True)


def _row(index: int, *, scheduled_ms: float = _SAMPLE_MS) -> dict[str, Any]:
    production = [_SAMPLE_MS] * _TRIALS
    scheduled = [scheduled_ms] * _TRIALS
    return {
        "shape_class": ("aligned", "ragged")[index % 2],
        "correctness": {"passed": True},
        "compile": {"digests": {"image": f"{index:064x}"}},
        "timing": {
            "production_samples_ms": production,
            "scheduled_samples_ms": scheduled,
            "non_regression_10pct": scheduled_ms <= _SAMPLE_MS * 1.10,
        },
    }


def _benchmark(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else [_row(0), _row(1)]
    regressed = not all(row["timing"]["non_regression_10pct"] for row in rows)
    return {
        "schema": "tessera.compiler.e2e_real4.x86_matmul.v1",
        "architecture": "zen5-avx512",
        "ratchet": {"kind": "production_non_regression", "limit": 1.10},
        "rows": rows,
        "verdict": "retain" if regressed else "promote",
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


def _calibration(*, cpu: int = _CPU) -> dict[str, Any]:
    windows = []
    for i in range(5):
        start = i * 300_000_000
        end = start + 200_000_000
        windows.append({"raw_start_ns": start, "raw_end_ns": end,
                        "tsc_start": int(start * _HZ / 1e9), "tsc_end": int(end * _HZ / 1e9),
                        "cpu": cpu})
    rates = [(w["tsc_end"] - w["tsc_start"]) * 1e9 / (w["raw_end_ns"] - w["raw_start_ns"])
             for w in windows]
    return {"frequency_hz": _HZ, "spread": 0.0, "rates_hz": rates, "windows": windows}


def _witness(row: dict[str, Any], *, tsc_scale: float = 1.0, image: str | None = None,
             env: str = "wsl2", measure_cpu: int = _CPU, gap_ns: int = 100_000) -> dict[str, Any]:
    from tessera.compiler.profiler_x86_clock import witness_sample

    timing = row["timing"]
    sampled_ns = int(1e6 * (sum(timing["production_samples_ms"]) + sum(timing["scheduled_samples_ms"])))
    raw_ns = sampled_ns + gap_ns
    start = 2_000_000_000
    window = {"raw_start_ns": start, "raw_end_ns": start + raw_ns,
              "tsc_start": 10, "tsc_end": 10 + int(raw_ns * _HZ / 1e9 * tsc_scale),
              "logical_cpu_start": measure_cpu, "logical_cpu_end": measure_cpu,
              "host_wall_ns": raw_ns + 1000}
    return witness_sample(_calibration(), window,
                          {"image": image or row["compile"]["digests"]["image"]},
                          execution_environment=env)


def _witnessed(rows: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    benchmark = _benchmark(rows)
    for row in benchmark["rows"]:
        row["timing_witness"] = _witness(row, **overrides)
    return benchmark


def _packet(benchmark: dict[str, Any], **timing: bool) -> dict[str, Any]:
    return build_x86_profiler_packet(
        benchmark=benchmark, timing_status={**_timing(), **timing}, cpu=_cpu(),
        environment=_environment(wsl=True), sampling=None)


def _reseal(packet: dict[str, Any]) -> dict[str, Any]:
    """Recompute both digests, as a forger would."""
    packet = copy.deepcopy(packet)
    packet["benchmark_sha256"] = digest_json(packet["benchmark"])
    packet.pop("packet_sha256", None)
    packet["packet_sha256"] = digest_json(packet)
    return packet


def test_wsl_zen5_packet_is_retain_not_promotion() -> None:
    packet = _packet(_benchmark())
    assert packet["admission_route"] == "profiler_correlated"
    assert packet["verdict"] == "retain"
    assert packet["eligible_for_promotion"] is False
    assert "WSL_CLOCK_DOMAIN" in packet["ineligibility_reasons"]
    assert "SYMBOL_SAMPLING_MISSING" in packet["ineligibility_reasons"]
    validate_x86_profiler_packet(packet)


def test_packet_digest_and_shape_coverage_fail_closed() -> None:
    packet = build_x86_profiler_packet(
        benchmark=_benchmark(), timing_status=_timing(), cpu=_cpu(),
        environment=_environment(wsl=False), sampling=None)
    tampered = copy.deepcopy(packet)
    tampered["benchmark"]["verdict"] = "reject"
    with pytest.raises(X86ProfilerPacketError, match="digest"):
        validate_x86_profiler_packet(tampered)

    benchmark = _benchmark([_row(0)])
    with pytest.raises(X86ProfilerPacketError, match="aligned and ragged"):
        build_x86_profiler_packet(
            benchmark=benchmark, timing_status=_timing(), cpu=_cpu(),
            environment=_environment(wsl=False), sampling=None)


def test_every_row_witnessed_takes_the_tsc_route_and_demotes_environment_tags() -> None:
    packet = _packet(_witnessed(), rdtscp_valid=False, clock_agreement_valid=False)
    assert packet["admission_route"] == "tsc_witness"
    assert packet["ineligibility_reasons"] == []
    assert packet["eligible_for_promotion"] is True
    gap_tags = {gap.partition(":")[0] for gap in packet["diagnostic_gaps"]}
    assert {"WSL_CLOCK_DOMAIN", "SYMBOL_SAMPLING_MISSING", "TIMING_PROOF_INCOMPLETE"} <= gap_tags
    validate_x86_profiler_packet(packet)


def test_a_regressed_benchmark_never_becomes_a_promote_packet() -> None:
    """Review P0: a benchmark whose own verdict is `retain` promoted on the
    tsc route, because only `reject` was read from the benchmark."""
    benchmark = _witnessed([_row(0), _row(1, scheduled_ms=_SAMPLE_MS * 1.5)])
    assert benchmark["verdict"] == "retain"
    packet = _packet(benchmark)
    assert packet["admission_route"] == "tsc_witness"
    assert packet["verdict"] == "retain" and packet["eligible_for_promotion"] is False
    validate_x86_profiler_packet(packet)

    # A stored flag that lies about its own samples is refused, before or after sealing.
    lying = copy.deepcopy(benchmark)
    lying["rows"][1]["timing"]["non_regression_10pct"] = True
    lying["verdict"] = "promote"
    with pytest.raises(X86ProfilerPacketError, match="disagrees with its own samples"):
        _packet(lying)
    forged = copy.deepcopy(packet)
    forged["benchmark"] = lying
    forged["verdict"] = "promote"
    forged["eligible_for_promotion"] = True
    with pytest.raises(X86ProfilerPacketError, match="disagrees with its own samples"):
        validate_x86_profiler_packet(_reseal(forged))


def test_avx512_visibility_still_blocks_on_the_tsc_route() -> None:
    packet = _packet(_witnessed(), avx512_visible=False)
    assert packet["admission_route"] == "tsc_witness"
    assert packet["eligible_for_promotion"] is False
    assert any("avx512_visible" in reason for reason in packet["ineligibility_reasons"])
    validate_x86_profiler_packet(packet)


@pytest.mark.parametrize("damage", [
    "missing_row", "disagreeing_tsc", "other_image", "migrated_cpu", "samples_not_bound",
])
def test_one_bad_row_keeps_the_profiler_route(damage: str) -> None:
    benchmark = _witnessed()
    row = benchmark["rows"][1]
    if damage == "missing_row":
        del row["timing_witness"]
    elif damage == "disagreeing_tsc":
        row["timing_witness"] = _witness(row, tsc_scale=1.10)
    elif damage == "other_image":
        row["timing_witness"] = _witness(row, image="f" * 64)
    elif damage == "migrated_cpu":
        row["timing_witness"] = _witness(row, measure_cpu=_CPU + 1)
    else:
        # The witnessed region is 25% longer than the samples it claims to
        # bracket: the verdict's numbers are not what the witness measured.
        row["timing_witness"] = _witness(row, gap_ns=10_000_000)
    packet = _packet(benchmark)
    assert packet["admission_route"] == "profiler_correlated"
    assert packet["eligible_for_promotion"] is False
    assert "WSL_CLOCK_DOMAIN" in packet["ineligibility_reasons"]
    validate_x86_profiler_packet(packet)


@pytest.mark.parametrize("forgery", [
    "scaled_samples", "stale_window", "garbage_calibration_digest", "moved_cpu",
])
def test_a_resealed_forgery_of_the_witness_is_refused(forgery: str) -> None:
    """Review P1: the validator trusted stored clock values and a self-declared
    calibration. Each forgery below was admitted before; resealing both digests
    must no longer help, because the route re-derives from stored integers."""
    packet = _packet(_witnessed())
    assert packet["admission_route"] == "tsc_witness"
    forged = copy.deepcopy(packet)
    row = forged["benchmark"]["rows"][0]
    tsc = row["timing_witness"]["clocks"]["tsc_cycles"]
    if forgery == "scaled_samples":
        row["timing"]["scheduled_samples_ms"] = [v * 0.1 for v in row["timing"]["scheduled_samples_ms"]]
    elif forgery == "stale_window":
        tsc["value"] *= 2
        tsc["provenance"]["calibrated_frequency_hz"] *= 2
    elif forgery == "garbage_calibration_digest":
        tsc["provenance"]["calibration_sha256"] = "0" * 64
    else:
        for interval in tsc["provenance"]["calibration"]["windows"]:
            interval["cpu"] = _CPU + 2
    with pytest.raises(X86ProfilerPacketError, match="admission route"):
        validate_x86_profiler_packet(_reseal(forged))


def test_validator_rederives_route_and_reason_split() -> None:
    packet = _packet(_witnessed())
    forged = copy.deepcopy(_packet(_benchmark()))
    forged["admission_route"] = "tsc_witness"
    with pytest.raises(X86ProfilerPacketError, match="admission route"):
        validate_x86_profiler_packet(forged)
    for field_name, value in (("diagnostic_gaps", []), ("ineligibility_reasons", ["VIRTUALIZED_HOST"])):
        moved = copy.deepcopy(packet)
        moved[field_name] = value
        with pytest.raises(X86ProfilerPacketError, match="reasons/gaps|has blockers"):
            validate_x86_profiler_packet(moved)


def test_v1_packets_are_read_only_under_the_subset_rule() -> None:
    """Review P2: the looser rule applies to the v1 schema, not to any packet
    that simply omits the route keys."""
    packet = _packet(_benchmark())
    stripped = copy.deepcopy(packet)
    del stripped["admission_route"], stripped["diagnostic_gaps"]
    stripped["ineligibility_reasons"] = []
    with pytest.raises(X86ProfilerPacketError):
        validate_x86_profiler_packet(_reseal(stripped))
    v1_with_route = copy.deepcopy(packet)
    v1_with_route["schema"] = "tessera.profiler_x86_packet.v1"
    with pytest.raises(X86ProfilerPacketError, match="predates admission routes"):
        validate_x86_profiler_packet(_reseal(v1_with_route))


def test_checked_in_princess_luna_packet_validates() -> None:
    """The committed Zen 5 packet re-validates off-host (no device, no rebuild).

    Its production and scheduled images are byte-identical, so its verdict is a
    parity check under WSL2, not a performance promotion.
    """
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "baselines"
            / "x86_zen5_profiler_packet_20260926_princess_luna.json")
    packet = json.loads(path.read_text(encoding="utf-8"))
    validate_x86_profiler_packet(packet)
    assert packet["schema"] == "tessera.profiler_x86_packet.v2"
    assert packet["admission_route"] == "tsc_witness"
    assert packet["environment"]["worktree_dirty"] is False
    assert packet["cpu"]["model_name"].startswith("AMD RYZEN AI MAX+ 395")
    stamp = packet["benchmark"]["runtime_library_build"]
    assert stamp["optimized"] is True and stamp["level"].startswith("O2")
    for row in packet["benchmark"]["rows"]:
        digests = row["compile"]["digests"]
        assert digests["image"] == digests["production_image"]
        assert row["timing_witness"]["artifact_digests"]["image"] == digests["image"]
