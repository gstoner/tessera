"""Host-free contracts for the matched gfx1201 MXFP4 benchmark."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as bench
from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case,
    _fragment_order,
    _logical_inputs,
    _parse_case,
)
from tessera.compiler import rocm_mxfp4 as mx


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "benchmarks/baselines/gfx1201_mxfp4_kstep_prefill_20260922/evidence.json"
)


def test_matched_input_uses_canonical_physical_layouts() -> None:
    case = Case("decode", 3, 32, 64)
    inputs = _logical_inputs(case)
    packed = inputs["packed_row_major"]
    codes = mx.unpack_e2m1_codes(packed)

    np.testing.assert_array_equal(
        _fragment_order(packed, case.n, case.k),
        mx.to_fragment_order(packed),
    )
    assert inputs["a"].shape == (case.m, case.k)
    assert packed.shape == (case.n, case.k // 2)
    assert codes.shape == (case.n, case.k)
    assert inputs["b_scale"].shape == (case.k // 32, case.n)


def test_matched_input_keeps_row_reference_fold_exact() -> None:
    inputs = _logical_inputs(Case("decode", 2, 32, 64))
    scales = inputs["b_scale"]
    reference = inputs["row_reference"]
    delta = reference[None, :].astype(np.int16) - scales.astype(np.int16)
    assert int(delta.min()) == 0
    assert int(delta.max()) <= 2


def test_case_parser_keeps_workload_separate_from_shape() -> None:
    assert _parse_case("prefill:256x5120x8704") == Case(
        "prefill", 256, 5120, 8704
    )


def test_matched_timing_alternates_engine_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = object()
    second = object()
    order: list[object] = []
    warmed: list[object] = []
    monkeypatch.setattr(
        bench, "_warmup", lambda _hip, engine, _count: warmed.append(engine)
    )

    def sample(_hip: object, engine: object, *, iterations: int) -> float:
        assert iterations == 3
        order.append(engine)
        return float(len(order))

    monkeypatch.setattr(bench, "_timed_sample", sample)
    measured = bench._measure_interleaved(
        object(), [first, second], warmup=2, trials=3, iterations=3
    )
    assert warmed == [first, second]
    assert order == [first, second, second, first, first, second]
    assert measured[first] == [1.0, 4.0, 5.0]
    assert measured[second] == [2.0, 3.0, 6.0]


def test_rdna4_barrier_counter_uses_archived_split_barrier_opcodes() -> None:
    assert bench._rdna4_workgroup_barrier_mnemonics() == (
        "s_barrier_signal",
        "s_barrier_signal_isfirst",
        "s_barrier_wait",
    )
    isa = """
        s_barrier_signal -1
        s_barrier_wait -1
        s_barrier_signal_isfirst -1
        s_get_barrier_state s0, -1
    """
    assert bench._count_rdna4_workgroup_barriers(isa) == 3


def test_production_packet_preserves_original_source_and_results() -> None:
    packet = json.loads(EVIDENCE.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_matched_benchmark.v1"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["architecture"] == "gfx1201"
    source = packet["source"]
    assert packet["timing_order"] == "alternating_interleaved_per_shape"
    assert source["revision"] == "39605e560415cd629bd5d1f1769cc9980e81f4ef"
    # Recorded before the 2026-09-24 producer relabel (pipeline_name only).
    # gfx1201_mxfp4_producer_relabel_20260924 proves the relabelled generator
    # builds the same timed kernels; do not relabel old GPU timings.
    assert source["generator_sha256"] == "abc96ab7746692768c5ef228a87a76968c8a34548faa66668f3ea00bff5ed0b5"
    # These September timings retain their actual recorder bytes. The October
    # explicit-stream extension has its own fresh source-bound graph packet.
    assert source["benchmark_sha256"] == "1bd38025923af1e60c16ea3e9621c5dc851919f90a8f379ad3d1d0d3f9991837"

    by_case: dict[str, list[dict[str, object]]] = {}
    for row in packet["rows"]:
        by_case.setdefault(row["case"], []).append(row)
        assert row["median_ms"] > 0
    assert len(by_case) == 4
    for case, rows in by_case.items():
        assert len({row["matched_output_sha256"] for row in rows}) == 1, case
        tessera = next(row for row in rows if row["engine"] == "tessera")
        expected_axis = "split_k" if case.startswith("decode") else "group_m"
        assert tessera["metadata"]["schedule"][expected_axis] == 8
        assert tessera["metadata"]["isa"]["wmma_fp8_fp8"] == 2
        expected_barriers = 2 if case.startswith("decode") else 4
        assert tessera["metadata"]["isa"]["s_barrier"] == expected_barriers
        assert tessera["metadata"]["resources"]["scratch_bytes"] == 0
        assert tessera["metadata"]["resources"]["spills"] is False


@pytest.mark.parametrize("name", [
    "radiance-fast-forward.json.gz", "radiance-fast-reverse.json.gz",
    "radiance-legacy-epilogue-forward.json.gz", "radiance-legacy-epilogue-reverse.json.gz",
])
def test_current_captured_packets_bind_the_benchmark_sources(name: str) -> None:
    path = ROOT / "benchmarks/baselines/gfx1201_folded_fragment_groups_20261015" / name
    packet = json.loads(gzip.decompress(path.read_bytes()))
    assert packet["schema"] == "tessera.rocm.folded_native_package.v1"
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["dirty_worktree"] is False
    assert packet["source_commit"] == "7ac37d9e2e65d0d0130f4dad2bd11029f663372e"
    for relative in (
        "benchmarks/rocm/benchmark_gfx1201_mxfp4_production.py",
        "benchmarks/rocm/record_gfx1201_folded_native_package.py",
        "benchmarks/rocm/folded_graph_windows.py",
    ):
        assert packet["source_sha256"][relative] == hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()
