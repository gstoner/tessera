#!/usr/bin/env python3
"""Require every active Tessera IR fixture to pass in a fleet lit lane."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path


EXPECTED_CODES = {"PASS", "UNSUPPORTED"}


def _fixture_name(test_name: str) -> str:
    return test_name.rsplit(" :: ", 1)[-1]


def active_fixtures(root: Path) -> set[str]:
    active: set[str] = set()
    for path in root.rglob("*.mlir"):
        if "// UNSUPPORTED: true" not in path.read_text():
            active.add(path.relative_to(root).as_posix())
    return active


def reconcile(root: Path, reports: Iterable[Path]) -> dict[str, object]:
    fixtures = active_fixtures(root)
    passed: set[str] = set()
    unexpected: list[dict[str, str]] = []
    lanes: dict[str, dict[str, int]] = {}
    for report in reports:
        payload = json.loads(report.read_text())
        counts: dict[str, int] = {}
        for result in payload.get("tests", []):
            code = str(result.get("code", "UNRESOLVED"))
            name = _fixture_name(str(result.get("name", "")))
            counts[code] = counts.get(code, 0) + 1
            if code == "PASS":
                passed.add(name)
            elif code not in EXPECTED_CODES:
                unexpected.append(
                    {"lane": report.name, "fixture": name, "code": code}
                )
        lanes[report.name] = counts
    return {
        "active": len(fixtures),
        "covered": len(fixtures & passed),
        "uncovered": sorted(fixtures - passed),
        "unexpected": unexpected,
        "lanes": lanes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--fixtures-root", type=Path, default=Path("tests/tessera-ir")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reconcile(args.fixtures_root, args.reports)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if result["uncovered"] or result["unexpected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
