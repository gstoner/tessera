from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/parse_ncu_resources.py"
SPEC = importlib.util.spec_from_file_location("nvidia_ncu_resources", PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_ncu_csv_normalizes_resources_spills_and_fingerprint():
    header = ("\"ID\",\"Kernel Name\",\"CC\",\"Block Size\",\"Grid Size\","
              "\"Metric Name\",\"Metric Value\"\n")
    rows = [
        '"0","kernel(float*)","12.0","256","257","Registers Per Thread","28"',
        '"0","kernel(float*)","12.0","256","257","Static Shared Memory Per Block","1024"',
        '"0","kernel(float*)","12.0","256","257","Theoretical Occupancy","100"',
        '"0","kernel(float*)","12.0","256","257","Achieved Occupancy","57.45"',
        '"0","kernel(float*)","12.0","256","257","Local Load Transactions","0"',
        '"0","kernel(float*)","12.0","256","257","Local Store Transactions","0"',
    ]
    parsed = mod.parse_csv(header + "\n".join(rows) + "\n")
    assert len(parsed) == 1
    row = parsed[0]
    assert row["registers_per_thread"] == 28
    assert row["static_shared_memory_bytes"] == 1024
    assert row["achieved_occupancy_pct"] == 57.45
    assert row["spill_evidence_complete"] is True
    assert row["spills_detected"] is False
    assert row["resource_fingerprint"].startswith("sha256:")


def test_ncu_csv_accepts_cuda_13_spill_request_counters():
    header = ("\"ID\",\"Kernel Name\",\"CC\",\"Block Size\",\"Grid Size\","
              "\"Metric Name\",\"Metric Value\"\n")
    rows = [
        '"0","kernel(float*)","12.0","64","4","Registers Per Thread","40"',
        '"0","kernel(float*)","12.0","64","4","Local Memory Spilling Requests","0"',
        '"0","kernel(float*)","12.0","64","4","Shared Memory Spilling Requests","0"',
    ]
    row = mod.parse_csv(header + "\n".join(rows) + "\n")[0]
    assert row["local_spill_requests"] == 0
    assert row["shared_spill_requests"] == 0
    assert row["spill_evidence_complete"] is True
    assert row["spills_detected"] is False
