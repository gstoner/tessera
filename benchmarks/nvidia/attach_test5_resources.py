"""Attach normalized route resources and cache/compiler evidence to baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _route(row: dict[str, Any]) -> str:
    if row.get("selected_route"):
        return str(row["selected_route"])
    mode = str(row.get("mode", ""))
    if mode in {"async_ring", "fused_paged_attention", "staged_paged_attention"}:
        return mode
    return mode.split(":", 1)[0]


def attach(payload: dict[str, Any], manifest: dict[str, Any],
           compiler_fingerprint: str) -> dict[str, Any]:
    rows = payload.get("rows", payload.get("runs", []))
    for row in rows:
        route = _route(row)
        detail = list(manifest.get("details", {}).get(route, []))
        row["selected_route"] = route
        row["compiler_fingerprint"] = row.get(
            "compiler_fingerprint", compiler_fingerprint)
        row["compile_state"] = row.get(
            "compile_state", "warm_after_preflight")
        row["cache_state"] = row.get("cache_state", "warm")
        row["resource_fingerprints"] = [
            item["resource_fingerprint"] for item in detail]
        row["resources"] = detail
        row["resource_evidence_complete"] = bool(detail) and all(
            item.get("spill_evidence_complete") for item in detail)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baselines", type=Path, nargs="+")
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.resources.read_text())
    version = subprocess.run(
        ["/usr/local/cuda/bin/nvcc", "--version"], check=True,
        capture_output=True, text=True).stdout
    fingerprint = "sha256:" + hashlib.sha256(version.encode()).hexdigest()
    for path in args.baselines:
        payload = attach(json.loads(path.read_text()), manifest, fingerprint)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
