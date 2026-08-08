from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.profiler_symbol_sampling import (
    PerfSample,
    SymbolSamplingError,
    build_symbol_sampling_artifact,
    ibs_capability,
    parse_nm_symbols,
    parse_perf_script,
    select_symbol,
    validate_symbol_sampling_artifact,
)


def test_nm_and_perf_records_correlate_by_dso_and_symbol(tmp_path: Path) -> None:
    binary = tmp_path / "kernel.so"
    binary.write_bytes(b"ELF-test-image")
    symbol = select_symbol(
        parse_nm_symbols("0000000000001130 0000000000000040 T zen5_kernel\n"),
        "zen5_kernel",
    )
    samples, rejected = parse_perf_script(
        "cycles:u 7f001150 zen5_kernel+0x20 (/tmp/kernel.so) 8\ncycles:u 7f001180 other_kernel (/tmp/kernel.so) 2\n"
    )
    artifact = build_symbol_sampling_artifact(
        binary=binary,
        build_id="abc123",
        target_symbol=symbol,
        event="cycles:u",
        samples=samples,
        rejected_lines=rejected,
        cpu={"vendor_id": "AuthenticAMD", "cpu_family": 26, "flags": ["avx512f"]},
        ibs={"eligible": False},
        perf_version="perf version test",
        perf_record_command=["perf", "record"],
        perf_returncode=0,
    )
    assert artifact["correlation"]["samples"] == 1
    assert artifact["correlation"]["coverage"] == pytest.approx(0.8)
    assert artifact["correlation"]["aslr_safe"] is True
    assert artifact["eligible_for_regression"] is True
    validate_symbol_sampling_artifact(artifact)


def test_sampling_rejects_ambiguous_or_unmapped_symbols() -> None:
    symbols = parse_nm_symbols("00000010 00000004 T duplicate\n00000020 00000004 T duplicate\n")
    with pytest.raises(SymbolSamplingError, match="exactly one"):
        select_symbol(symbols, "duplicate")


def test_ibs_requires_amd_family_26_and_advertised_event() -> None:
    zen5 = ibs_capability(
        cpu={"vendor_id": "AuthenticAMD", "cpu_family": 26, "flags": ["avx512f"]},
        perf_list_text="ibs_op// [Kernel PMU event]\nibs_fetch//",
    )
    assert zen5["eligible"] is True
    assert zen5["events"] == {"ibs_op": True, "ibs_fetch": True}
    wrong_family = ibs_capability(
        cpu={"vendor_id": "AuthenticAMD", "cpu_family": 25, "flags": ["avx512f"]},
        perf_list_text="ibs_op//",
    )
    assert wrong_family["eligible"] is False


def test_sampling_artifact_alone_cannot_claim_promotion(tmp_path: Path) -> None:
    binary = tmp_path / "kernel.so"
    binary.write_bytes(b"image")
    artifact = build_symbol_sampling_artifact(
        binary=binary,
        build_id="build-id",
        target_symbol=select_symbol(parse_nm_symbols("00000010 00000004 T kernel\n"), "kernel"),
        event="cycles:u",
        samples=[PerfSample(0x1000, "kernel", str(binary), 1, "cycles:u")],
        rejected_lines=[],
        cpu={},
        ibs={},
        perf_version="test",
        perf_record_command=[],
        perf_returncode=0,
    )
    artifact["eligible_for_promotion"] = True
    with pytest.raises(SymbolSamplingError, match="alone"):
        validate_symbol_sampling_artifact(artifact)
