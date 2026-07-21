#!/usr/bin/env python3
"""Measure E2E-SPINE-3 report/packet validation overhead.

This is an infrastructure benchmark, not backend performance evidence.  It
proves that evidence validation happens after measurement and is cheap enough
to keep completely outside device-event and end-to-end timing intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from tessera.compiler.e2e_fleet import (
    REPORT_SCHEMA,
    load_fixture_corpus,
    seal_packet,
    validate_backend_report,
    validate_packet,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _report() -> dict:
    fixture = load_fixture_corpus()["softmax-f32-2x2-extreme-v1"]
    image = _digest("benchmark:image")
    descriptor = _digest("benchmark:descriptor")
    cache_key = _digest("benchmark:cache")
    return {
        "schema": REPORT_SCHEMA,
        "target": "validation_host",
        "device": {"exact": True, "identity": "validation-benchmark-only"},
        "source_commit": "0" * 40,
        "toolchain_fingerprint": "python-validation-benchmark",
        "scope": ["softmax"],
        "required_timing_domains": ["kernel_wall", "end_to_end"],
        "fixtures": [{
            "fixture_id": fixture["fixture_id"],
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": fixture["oracle"],
            "image_digest": image,
            "descriptor_digest": descriptor,
        }],
        "cache_proofs": [{
            "fixture_id": fixture["fixture_id"],
            "cold": {
                "compile_state": "cold", "cache_key": cache_key,
                "image_digest": image, "descriptor_digest": descriptor,
            },
            "warm": {
                "compile_state": "warm_cache", "cache_key": cache_key,
                "image_digest": image, "descriptor_digest": descriptor,
            },
        }],
        "benchmarks": [
            {
                "family": "softmax", "route": "benchmark_fixture",
                "timing_domain": domain, "median_ns": 100.0,
                "run_medians_ns": [99.0, 101.0],
                "stability_limit_pct": 5.0, "stable": True,
                "selected": True,
                "repetitions": 10, "warmups": 1, "discard_first": True,
                "resource_fingerprint": _digest("benchmark:resources"),
            }
            for domain in ("kernel_wall", "end_to_end")
        ],
    }


def _measure(callable_, repetitions: int) -> dict[str, float | int]:
    callable_()  # discard import/cache/first-filesystem-use effects
    samples: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        callable_()
        samples.append(time.perf_counter_ns() - start)
    return {
        "repetitions": repetitions,
        "median_ns": statistics.median(samples),
        "minimum_ns": min(samples),
        "maximum_ns": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    report = _report()
    with tempfile.TemporaryDirectory(prefix="tessera-e2e-fleet-") as temp:
        packet = Path(temp)
        (packet / "report.json").write_text(json.dumps(report), encoding="utf-8")
        seal_packet(packet)
        result = {
            "schema": "tessera.e2e-evidence-validation-benchmark.v1",
            "timing_scope": "host validation only; excluded from backend timing",
            "discard_first": True,
            "report_validation": _measure(
                lambda: validate_backend_report(report), args.reps,
            ),
            "sealed_packet_validation": _measure(
                lambda: validate_packet(packet), args.reps,
            ),
        }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
