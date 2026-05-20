"""Render the operator-benchmark coverage dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tessera.compiler.operator_benchmarks_coverage import render_markdown


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "audit" / "generated" / "operator_benchmarks_coverage.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
