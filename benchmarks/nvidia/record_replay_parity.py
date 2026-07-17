"""Record the wider exact-device ReplaySSM async-ring parity matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_replay_parity.json"
RESOURCES = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"
NOISE = .03


def record(*, reps: int = 20) -> dict[str, Any]:
    from benchmarks.nvidia.benchmark_serving import _replay_row
    from tessera import runtime as rt

    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("ReplaySSM parity requires exact sm_120")
    resource = json.loads(RESOURCES.read_text())["details"]["async_ring"]
    rows = []
    for shape in ("1x32x16", "1x64x64", "1x128x64",
                  "1x128x128", "4x64x64"):
        for tokens, chunk, slots in ((16, 4, 2), (64, 16, 4)):
            measured = _replay_row(
                shape, tokens, chunk, slots, reps * 8,
                condition_clock=True, retain_samples=True,
                condition_every=4)
            raw_device = measured.pop("device_samples_ms_per_token")
            raw_e2e = measured.pop("end_to_end_samples_ms_per_token")
            device_samples = [statistics.median(raw_device[i:i + 4])
                              for i in range(0, len(raw_device), 4)]
            e2e_samples = [statistics.median(raw_e2e[i:i + 4])
                           for i in range(0, len(raw_e2e), 4)]
            runs = []
            for run_id in (1, 2):
                device_batches = device_samples[run_id - 1::2]
                e2e_batches = e2e_samples[run_id - 1::2]
                runs.append({
                    "run": run_id,
                    "device_event_ms_per_token": statistics.median(
                        device_batches),
                    "end_to_end_ms_per_token": statistics.median(e2e_batches),
                    "max_abs_error": measured["max_abs_error"],
                    "clock_condition_device_ms": measured[
                        "clock_condition_device_ms"],
                    "device_batch_medians_ms_per_token": device_batches,
                    "end_to_end_batch_medians_ms_per_token": e2e_batches,
                    "sampling_order": "route_sample_odd_even_interleaved",
                })
            stable = {}
            for domain in ("device_event_ms_per_token", "end_to_end_ms_per_token"):
                a, b = (run[domain] for run in runs)
                stable[domain] = abs(a - b) / max(a, b) <= NOISE
            traffic = measured
            rows.append({
                "shape": shape, "tokens": tokens, "chunk": chunk,
                "async_slots": slots, "ring_waves": traffic["ring_waves"],
                "runs": runs,
                "device_stable": stable["device_event_ms_per_token"],
                "end_to_end_stable": stable["end_to_end_ms_per_token"],
                "stable": all(stable.values()),
                "state_bytes_per_token": traffic["state_bytes_per_token"],
                "summary_state_bytes_per_token": traffic["summary_state_bytes_per_token"],
                "state_traffic_ratio": traffic["state_traffic_ratio"],
                "resource_fingerprints": [r["resource_fingerprint"] for r in resource],
                "resources": resource, "resource_evidence_complete": bool(resource),
            })
    version = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"],
                             check=True, capture_output=True, text=True).stdout
    return {
        "schema": "tessera.nvidia.replay-parity.v1",
        "device": "nvidia:sm_120", "noise_policy": NOISE,
        "compiler_fingerprint": "sha256:" + hashlib.sha256(version.encode()).hexdigest(),
        "transition_proofs": ["long_decode", "flush", "rollback",
                              "speculative_rejection", "block_submit",
                              "ordered_ring", "backpressure", "teardown"],
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record(reps=args.reps)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({len(result['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
