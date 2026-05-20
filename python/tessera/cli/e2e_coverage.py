"""``tessera-e2e-coverage`` — CLI for the E2E op coverage audit.

Three subcommands (all flag-style for consistency with the other
audit CLIs):

  * ``--render`` — write ``docs/audit/generated/e2e_op_coverage.md``
    from the live coverage data.
  * ``--check`` — compare the generated doc against
    ``render_markdown()`` and exit non-zero on drift.  CI gate.
  * ``--summary`` — print the tier-count rollup as one line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tessera.compiler.e2e_coverage import (
    render_markdown,
    status_counts,
    write_doc,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DOC = (
    _REPO_ROOT / "docs" / "audit" / "generated" / "e2e_op_coverage.md"
)


def _cmd_render(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else _GENERATED_DOC
    written = write_doc(out)
    print(f"wrote {written}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if not _GENERATED_DOC.exists():
        print(
            f"FAIL: {_GENERATED_DOC} missing — run "
            f"`python -m tessera.cli.e2e_coverage --render` first.",
            file=sys.stderr,
        )
        return 1
    on_disk = _GENERATED_DOC.read_text(encoding="utf-8")
    rendered = render_markdown()
    if on_disk != rendered:
        print(
            f"FAIL: {_GENERATED_DOC} is out of date.\n"
            f"Run `python -m tessera.cli.e2e_coverage --render` to update.",
            file=sys.stderr,
        )
        return 1
    counts = status_counts()
    print(f"[e2e_coverage] clean — {counts}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    counts = status_counts()
    total = sum(counts.values())
    parts = [f"{tier}={n}" for tier, n in counts.items()]
    print(f"E2E coverage ({total} ops): " + " ".join(parts))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-e2e-coverage",
        description=(
            "Audit the E2E compiler-stack coverage for every op in "
            "tessera.compiler.op_catalog.OP_SPECS."
        ),
    )
    p.add_argument(
        "--render", action="store_true",
        help="Regenerate docs/audit/generated/e2e_op_coverage.md.",
    )
    p.add_argument(
        "--check", action="store_true",
        help="CI gate: fail if the generated doc is out of date.",
    )
    p.add_argument(
        "--summary", action="store_true",
        help="Print the tier-count rollup as one line.",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output path for --render "
            "(default: docs/audit/generated/e2e_op_coverage.md)"
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.render:
        return _cmd_render(args)
    if args.check:
        return _cmd_check(args)
    if args.summary:
        return _cmd_summary(args)
    return _cmd_summary(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
