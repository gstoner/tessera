"""Drift gate for the committed NVIDIA-TEST-5 evidence matrix."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
BASE = ROOT / "benchmarks/baselines"


def _load(name: str):
    return json.loads((BASE / name).read_text())


def test_test5_corpus_has_stable_dual_domain_shape_matrix():
    rows = [r for r in _load("autotune_corpus.json")["records"]
            if r["device"] == "nvidia:sm_120" and r["op"] == "matmul"
            and tuple(r.get("bucket") or ()) in {
                (512, 512, 512), (128, 256, 64), (128, 512, 64)}]
    assert {r["timing"] for r in rows} == {"end_to_end", "device"}
    assert {(tuple(r["evidence"].get("workload_shape", r["bucket"])),
             r["dtype"]) for r in rows} >= {
        ((512, 512, 512), "float16"),
        ((128, 256, 64), "float16"),
        ((127, 259, 63), "float16"),
    }
    assert all(r["evidence"]["stable_runs"] == 2 for r in rows)
    assert all(r["evidence"]["resource_fingerprints"] for r in rows)

    # ELIGIBILITY MUST FOLLOW STABILITY, rather than every row being stable.
    #
    # `all(stable_winner)` was the old assertion and it is not a property the
    # hardware has. `finalize_test5_corpus` marks a row stable only when two
    # independent runs pick the same winner, and at `bfloat16 [128, 256, 64]`
    # device timing they do not -- that shape sits at the launch-overhead floor
    # where the corpus separately measures a 9.92% margin against 102.96%
    # noise. Two mechanisms built years apart, the finalizer's two-run
    # agreement and `MeasureRecord.separation`, independently say the ranking
    # is not real.
    #
    # So the invariant worth holding is not "every row is reproducible" but
    # "the corpus never offers a winner it cannot reproduce": eligible implies
    # stable. A row that is honest about being unstable is doing its job.
    for row in rows:
        evidence = row["evidence"]
        if evidence["selector_eligible"]:
            assert evidence["stable_winner"], (
                f"{row['dtype']} {row['bucket']} {row['timing']} is offered to "
                "the selector while its two runs disagreed on the winner")
    # ...and the matrix must not degrade to all-unstable without anyone noticing.
    assert any(r["evidence"]["stable_winner"] for r in rows)


def test_test5_standalone_baselines_have_dual_timing_and_resources():
    backward = _load("nvidia_sm120_attention_backward.json")["rows"]
    assert {r["timing_domain"] for r in backward} == {
        "end_to_end", "device_event"}
    assert all(r["resource_evidence_complete"] for r in backward)

    reduction = _load("nvidia_sm120_reduction_transport.json")["rows"]
    assert {r["timing_domain"] for r in reduction} == {
        "end_to_end", "device_event"}
    assert {r["op"] for r in reduction} >= {
        "reduction_sum", "reduction_mean", "reduction_max",
        "moe_dispatch", "moe_combine", "grouped_gemm"}
    assert all(r["resource_evidence_complete"] for r in reduction)

    serving = _load("nvidia_sm120_serving.json")["runs"]
    assert {r["op"] for r in serving} == {
        "ssm_replay_decode", "paged_kv_decode"}
    assert all(r["latency_ms"] > 0 and r["device_latency_ms"] > 0
               for r in serving)
    assert all(r["resource_evidence_complete"] for r in serving)


def test_test5_resource_manifest_has_every_selected_route_and_spill_state():
    manifest = _load("nvidia_sm120_test5_route_resources.json")
    required = {
        "nvidia_mma_gemm_shipped", "nvidia_tile_matmul_direct",
        "nvidia_tile_matmul_shared", "nvidia_generic_cuda",
        "nvidia_mma_fused", "nvidia_mma_attn",
        "generated_atomic_vjp", "generated_row_reduce", "generated_gather",
        "generated_combine", "generated_grouped", "fused_paged_attention",
        "staged_paged_attention", "async_ring", "direct",
    }
    assert required <= set(manifest["routes"])
    assert all(item["spill_evidence_complete"]
               for route in required for item in manifest["details"][route])
    assert manifest["details"]["generated_atomic_vjp"][0]["spills_detected"]


def test_cuda_parity_gemm_matrix_retains_exact_cases_and_candidate_resources():
    payload = _load("nvidia_sm120_gemm_schedule_matrix.json")
    rows = payload["rows"]
    assert payload["schema"] == "tessera.nvidia.gemm-schedule-matrix.v1"
    assert payload["method"]["runs"] == 2
    assert payload["method"]["selector_changed"] is False
    assert {row["timing"] for row in rows} == {"device", "end_to_end"}
    assert {row["case"].split(":")[1] for row in rows
            if row["op"] == "matmul"} >= {"square", "rectangular", "ragged"}
    assert {row["case"].split(":")[1] for row in rows
            if row["op"] == "fused_region"} >= {"none", "relu", "gelu", "silu"}
    assert all(row["stable"] and row["selector_eligible"] for row in rows)
    assert all(len(row["runs"]) == 2 for row in rows)
    assert all(all(fingerprints for fingerprints in
                   row["candidate_resource_fingerprints"].values())
               for row in rows)
