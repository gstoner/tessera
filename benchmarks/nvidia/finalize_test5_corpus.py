"""Merge two fresh NVIDIA measurement runs with stability/resource evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    bucket = row.get("bucket")
    return (row["device"], row["target"], row["op"],
            tuple(bucket) if bucket is not None else None,
            row["dtype"], row.get("timing", "end_to_end"))


def merge(base: dict[str, Any], first: dict[str, Any], second: dict[str, Any],
          resources: dict[str, Any], *, noise_fraction: float = 0.03) -> dict[str, Any]:
    """Replace NVIDIA rows only when both fresh runs agree on the winner."""
    a = {_key(r): r for r in first.get("records", [])}
    b = {_key(r): r for r in second.get("records", [])}
    routes = resources.get("routes", {})
    merged = {_key(r): dict(r) for r in base.get("records", [])}
    for key in sorted(a.keys() & b.keys(), key=str):
        left, right = a[key], dict(b[key])
        raw_winners = [left["winner"], right["winner"]]
        def near(row: dict[str, Any]) -> set[str]:
            candidates = {str(k): float(v) for k, v in row["candidates"].items()}
            floor = min(candidates.values())
            return {name for name, value in candidates.items()
                    if value <= floor * (1.0 + noise_fraction)}
        consensus = near(left) & near(right)
        stable = bool(consensus)
        if stable:
            winner = min(consensus, key=lambda name: (
                float(left["candidates"][name]) +
                float(right["candidates"][name]), name))
            right["winner"] = winner
            right["latency_ms"] = float(right["candidates"][winner])
        route_resources = list(routes.get(right["winner"], []))
        evidence = dict(right.get("evidence", {}))
        evidence.update({
            "stable_runs": 2,
            "run_winners": raw_winners,
            "noise_fraction": noise_fraction,
            "near_winner_consensus": sorted(consensus),
            "stable_winner": stable,
            "selector_eligible": bool(stable and route_resources),
            "resource_fingerprints": route_resources,
        })
        right["evidence"] = evidence
        # Unstable evidence is retained for audit but cannot displace a stable
        # selector row already committed for the same key.
        if stable or key not in merged:
            merged[key] = right
        elif key in merged:
            prior = dict(merged[key])
            prior["evidence"] = evidence
            merged[key] = prior
    return {"version": 3,
            "records": sorted(merged.values(), key=lambda r: str(_key(r)))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-fraction", type=float, default=0.03)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text())
    args.output.write_text(json.dumps(merge(
        load(args.base), load(args.first), load(args.second),
        load(args.resources), noise_fraction=args.noise_fraction),
        indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
