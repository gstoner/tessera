"""Finalize two-run TF32/FP8 native-vs-composed selector evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DTYPES = {"f32", "fp8_e4m3", "fp8_e5m2"}
OPS = {"fused_region", "attention", "gated_matmul"}


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["device"], row["target"], row["op"],
            tuple(row.get("bucket") or ()), row["dtype"], row["timing"])


def _near(row: dict[str, Any], noise: float) -> set[str]:
    values = {name: float(value) for name, value in row["candidates"].items()}
    floor = min(values.values())
    return {name for name, value in values.items()
            if value <= floor * (1.0 + noise)}


def finalize(first: dict[str, Any], second: dict[str, Any],
             resources: dict[str, Any], noise: float = 0.03
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a = {_key(row): row for row in first["records"]
         if row["dtype"] in DTYPES and row["op"] in OPS}
    b = {_key(row): row for row in second["records"]
         if row["dtype"] in DTYPES and row["op"] in OPS}
    resource_map: dict[str, list[dict[str, Any]]] = {}
    for row in resources["rows"]:
        resource_map.setdefault(row["candidate"], []).append(row)
    identities = sorted({key[:-1] for key in a.keys() & b.keys()}, key=str)
    rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    for identity in identities:
        domains: dict[str, dict[str, Any]] = {}
        for timing in ("end_to_end", "device"):
            key = (*identity, timing)
            if key not in a or key not in b:
                continue
            left, right = a[key], b[key]
            consensus = _near(left, noise) & _near(right, noise)
            domains[timing] = {
                "run_winners": [left["winner"], right["winner"]],
                "near_winner_consensus": sorted(consensus),
                "runs": [left["candidates"], right["candidates"]],
            }
        cross = (set(domains.get("end_to_end", {}).get(
            "near_winner_consensus", [])) & set(domains.get("device", {}).get(
                "near_winner_consensus", [])))
        winner = None
        if cross:
            winner = min(cross, key=lambda name: (
                sum(float(run[name]) for domain in domains.values()
                    for run in domain["runs"]), name))
        linked = resource_map.get(winner or "", [])
        eligible = bool(winner and linked)
        dev, target, op, bucket, dtype = identity
        shape = None
        for timing in ("end_to_end", "device"):
            evidence = b.get((*identity, timing), {}).get("evidence", {})
            if evidence.get("workload_shape"):
                shape = evidence["workload_shape"]
                break
        shape = shape or list(bucket)
        rows.append({
            "device": dev, "target": target, "op": op,
            "shape_bucket": list(bucket), "workload_shape": shape,
            "dtype": dtype, "winner": winner,
            "stable_runs": 2, "noise_fraction": noise,
            "timing_domain_consensus": bool(cross),
            "selector_eligible": eligible,
            "selector_promoted": eligible,
            "timings": domains,
            "resources": linked,
            "resource_fingerprints": [row["resource_fingerprint"]
                                      for row in linked],
        })
        if eligible:
            for timing, domain in domains.items():
                source = dict(b[(*identity, timing)])
                source["winner"] = winner
                source["latency_ms"] = float(source["candidates"][winner])
                source["evidence"] = {
                    **source.get("evidence", {}),
                    "stable_runs": 2,
                    "noise_fraction": noise,
                    "run_winners": domain["run_winners"],
                    "near_winner_consensus": domain["near_winner_consensus"],
                    "timing_domain_consensus": True,
                    "selector_eligible": True,
                    "resource_fingerprints": [row["resource_fingerprint"]
                                              for row in linked],
                }
                corpus_rows.append(source)
    payload = {
        "schema": "tessera.nvidia.lowp-native-routes.v1",
        "device": "nvidia:sm_120",
        "method": {"runs": 2, "noise_fraction": noise,
                   "timing_domains": ["end_to_end", "device"],
                   "promotion_requires_cross_domain_consensus": True},
        "rows": rows,
        "selector_promotions": sum(row["selector_promoted"] for row in rows),
    }
    return payload, corpus_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-corpus", type=Path)
    parser.add_argument("--corpus-output", type=Path)
    parser.add_argument("--noise-fraction", type=float, default=0.03)
    args = parser.parse_args(argv)
    load = lambda path: json.loads(path.read_text())
    payload, promoted = finalize(
        load(args.first), load(args.second), load(args.resources),
        args.noise_fraction)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.base_corpus or args.corpus_output:
        if not (args.base_corpus and args.corpus_output):
            parser.error("--base-corpus and --corpus-output must be used together")
        corpus = load(args.base_corpus)
        indexed = {_key(row): row for row in corpus["records"]}
        indexed.update({_key(row): row for row in promoted})
        corpus["records"] = sorted(indexed.values(), key=lambda row: str(_key(row)))
        args.corpus_output.write_text(
            json.dumps(corpus, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; promotions={payload['selector_promotions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
