"""``tessera-apple-target-map`` — CLI for the Apple target map.

Three flag-style subcommands matching the rest of the audit CLIs:

  * ``--render`` — write ``docs/audit/generated/apple_target_map.md``.
  * ``--check`` — fail if the on-disk doc is stale vs ``render_markdown()``.
  * ``--summary`` — print per-family counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tessera.compiler.apple_target_map import (
    all_rows, render_markdown, write_doc,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DOC = (
    _REPO_ROOT / "docs" / "audit" / "generated" / "apple_target_map.md"
)


def _cmd_render(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else _GENERATED_DOC
    written = write_doc(out)
    print(f"wrote {written}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if not _GENERATED_DOC.exists():
        print(
            f"FAIL: {_GENERATED_DOC} missing — "
            f"run `python -m tessera.cli.apple_target_map --render` first.",
            file=sys.stderr,
        )
        return 1
    on_disk = _GENERATED_DOC.read_text(encoding="utf-8")
    rendered = render_markdown()
    if on_disk != rendered:
        print(
            f"FAIL: {_GENERATED_DOC} is out of date.\n"
            f"Run `python -m tessera.cli.apple_target_map --render` to update.",
            file=sys.stderr,
        )
        return 1
    print(f"[apple_target_map] clean — {len(all_rows())} rows")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    rows = all_rows()
    by_family: dict[str, int] = {}
    fused_gpu = 0
    acc_cpu = 0
    for r in rows:
        by_family[r.family] = by_family.get(r.family, 0) + 1
        if r.gpu_status == "fused":
            fused_gpu += 1
        if r.cpu_execution_kind == "accelerate_native":
            acc_cpu += 1
    parts = [f"{f}={n}" for f, n in sorted(by_family.items())]
    print(
        f"Apple target map ({len(rows)} rows): " + " ".join(parts)
        + f" | apple_gpu fused={fused_gpu}"
        + f" | apple_cpu accelerate_native={acc_cpu}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-apple-target-map",
        description=(
            "Unified per-op view of apple_cpu vs apple_gpu coverage "
            "across capabilities, backend_manifest, and driver."
        ),
    )
    p.add_argument(
        "--render", action="store_true",
        help="Regenerate docs/audit/generated/apple_target_map.md.",
    )
    p.add_argument(
        "--check", action="store_true",
        help="CI gate: fail if the generated doc is out of date.",
    )
    p.add_argument(
        "--summary", action="store_true",
        help="Print per-family counts as one line.",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output path for --render "
            "(default: docs/audit/generated/apple_target_map.md)"
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
