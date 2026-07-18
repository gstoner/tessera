from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PATH = (Path(__file__).parents[2] / "benchmarks/nvidia"
        / "finalize_low_precision_native_routes.py")
SPEC = importlib.util.spec_from_file_location("lowp_finalize", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _run(end_winner="native", device_winner="native"):
    records = []
    for timing, winner in (("end_to_end", end_winner),
                           ("device", device_winner)):
        records.append({
            "device": "nvidia:sm_120", "target": "nvidia",
            "op": "attention", "bucket": [128, 128, 64, 64],
            "dtype": "fp8_e4m3", "timing": timing, "winner": winner,
            "latency_ms": 1.0,
            "candidates": {"native": 1.0, "composed": 1.2},
            "evidence": {"workload_shape": [128, 128, 64, 64]},
        })
    return {"records": records}


def test_promotion_requires_two_runs_both_domains_and_resources():
    resources = {"rows": [{"candidate": "native",
                            "resource_fingerprint": "sha256:n"}]}
    payload, corpus = MOD.finalize(_run(), _run(), resources)
    assert payload["selector_promotions"] == 1
    assert payload["rows"][0]["timing_domain_consensus"] is True
    assert len(corpus) == 2


def test_domain_disagreement_or_missing_resources_blocks_promotion():
    disagree = _run(device_winner="composed")
    # Make the device measurements genuinely prefer composed beyond 3%.
    disagree["records"][1]["candidates"] = {"native": 1.2, "composed": 1.0}
    payload, corpus = MOD.finalize(_run(), disagree, {"rows": []})
    assert payload["selector_promotions"] == 0
    assert corpus == []


def test_committed_native_route_ratchet_has_dual_domain_resources_and_guards():
    root = Path(__file__).parents[2]
    routes = json.loads((root / "benchmarks/baselines"
                         / "nvidia_sm120_low_precision_native_routes.json").read_text())
    resources = json.loads((root / "benchmarks/baselines"
                            / "nvidia_sm120_low_precision_native_resources.json").read_text())
    assert routes["selector_promotions"] == 11
    assert len(routes["rows"]) == 18
    assert all(set(row["timings"]) == {"end_to_end", "device"}
               for row in routes["rows"])
    promoted = [row for row in routes["rows"] if row["selector_promoted"]]
    assert all(row["timing_domain_consensus"] and row["resource_fingerprints"]
               for row in promoted)
    assert all(not row["selector_promoted"] for row in routes["rows"]
               if row["op"] == "attention" and row["workload_shape"][1] == 512)
    assert len(resources["rows"]) == 12
    known = {row["resource_fingerprint"] for row in resources["rows"]}
    assert {fingerprint for row in promoted
            for fingerprint in row["resource_fingerprints"]} <= known
    assert all(row["spill_evidence_complete"] and not row["spills_detected"]
               for row in resources["rows"])
    assert {family.split(".")[0] for row in resources["rows"]
            for family in row["sass_instruction_families"]} == {"HMMA", "QMMA"}
