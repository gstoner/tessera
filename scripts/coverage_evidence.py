#!/usr/bin/env python3
"""Generate revision-bound coverage evidence for CI or local inspection."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from validation_tree import snapshot
from tessera.compiler import generated_docs as gd


def export(directory: Path, root: Path) -> Path:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("coverage output directory must be empty")
    before = snapshot(root)
    expected = os.environ.get("GITHUB_SHA")
    if expected and before["head"] != expected:
        raise ValueError("coverage checkout differs from GITHUB_SHA")
    directory.mkdir(parents=True, exist_ok=True)
    doc = replace(gd.get("test_coverage"), committed=True,
                  md_path=directory / "test_coverage.md",
                  csv_path=directory / "test_coverage.csv")
    paths = gd.write(doc)
    error = gd.check(doc)
    if error:
        raise ValueError(error)
    if snapshot(root) != before:
        raise ValueError("coverage source tree changed during generation")
    manifest = {
        "schema": "tessera.coverage-evidence.v1",
        "source_commit": before["head"],
        "source_tree_sha256": hashlib.sha256(
            json.dumps(before["files"], sort_keys=True).encode()).hexdigest(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "generator": "tessera.compiler.generated_docs:test_coverage",
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope": "static test-reference inventory; not executed-test or device proof",
    }
    output = directory / "manifest.json"
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.output, Path(__file__).resolve().parents[1]))
