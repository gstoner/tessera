"""Validate NVIDIA autotune fingerprints and record cold/warm cache behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_autotune_reproducibility.json"
CORPUS = ROOT / "benchmarks/baselines/autotune_corpus.json"


def _fingerprints(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "resource_fingerprint" and isinstance(item, str):
                found.add(item)
            found.update(_fingerprints(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_fingerprints(item))
    return found


def _resource_fingerprints() -> set[str]:
    found = set()
    for path in (ROOT / "benchmarks/baselines").glob("nvidia*resource*.json"):
        found.update(_fingerprints(json.loads(path.read_text())))
    return found


def record() -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda  # noqa: F401 - registers plugin
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.kernel_cache import KernelCache, build
    from tessera.compiler.emit.kernel_emitter import SpecPolicy
    from tessera.compiler.fusion import FusedRegion

    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("autotune reproducibility requires exact sm_120")
    nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], check=True,
                          capture_output=True, text=True).stdout
    compiler = "sha256:" + hashlib.sha256(nvcc.encode()).hexdigest()
    builds = []
    keys = set()
    for run_id in (1, 2):
        cache = KernelCache()
        start = time.perf_counter_ns()
        cold = build(FusedRegion(("relu",)), "nvidia", SpecPolicy.DYNAMIC,
                     dtype="f32", dims=(127, 259, 63), cache=cache)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        warm = build(FusedRegion(("relu",)), "nvidia", SpecPolicy.DYNAMIC,
                     dtype="f32", dims=(127, 259, 63), cache=cache)
        warm_ms = (time.perf_counter_ns() - start) / 1e6
        assert cold is warm and cache.hits == 1 and cache.misses == 1
        keys.add(cold.key)
        builds.append({
            "run": run_id, "cold_compile_ms": cold_ms,
            "warm_cache_lookup_ms": warm_ms, "cache_hits": cache.hits,
            "cache_misses": cache.misses, "kernel_cache_key": cold.key,
            "artifact_present": bool(cold.artifact),
        })
    assert len(keys) == 1

    payload = json.loads(CORPUS.read_text())
    known_resources = _resource_fingerprints()
    eligible = [row for row in payload["records"]
                if row["target"] == "nvidia"
                and row.get("evidence", {}).get("selector_eligible") is True]
    admitted = 0
    for row in eligible:
        evidence = row["evidence"]
        resources = evidence.get("resource_fingerprints", [])
        if not resources or not set(resources).issubset(known_resources):
            raise RuntimeError("autotune row references a stale resource fingerprint")
        policy = {
            "device": "nvidia:sm_120", "timing": row["timing"],
            "compiler_fingerprint": compiler,
            "resource_fingerprints": resources,
            "selector_eligible": True,
        }
        cache = at.MeasureCache()
        admitted += cache.load_dict(
            {"version": payload["version"], "records": [row]},
            required_evidence=policy)
        if cache.size != 1:
            raise RuntimeError("selector-eligible autotune row failed strict admission")

    sample = eligible[0]
    sample_resources = sample["evidence"]["resource_fingerprints"]
    stale_rejections = {}
    base = {
        "device": "nvidia:sm_120", "timing": sample["timing"],
        "compiler_fingerprint": compiler,
        "resource_fingerprints": sample_resources,
        "selector_eligible": True,
    }
    mutations = {
        "device": "nvidia:sm_999", "timing": "invalid-domain",
        "compiler_fingerprint": "sha256:stale-compiler",
        "resource_fingerprints": ["sha256:stale-resource"],
    }
    for field, stale in mutations.items():
        cache = at.MeasureCache()
        stale_rejections[field] = cache.load_dict(
            {"version": payload["version"], "records": [sample]},
            required_evidence={**base, field: stale}) == 0

    return {
        "schema": "tessera.nvidia.autotune-reproducibility.v1",
        "device": "nvidia:sm_120", "compiler_fingerprint": compiler,
        "kernel_cache_reproducible": len(keys) == 1,
        "builds": builds, "strict_records_considered": len(eligible),
        "strict_records_admitted": admitted,
        "stale_rejections": stale_rejections,
        "selector_changed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = record()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
