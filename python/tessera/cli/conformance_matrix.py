"""CLI entry point for the op×target conformance matrix dashboard.

Usage::

    python -m tessera.cli.conformance_matrix --render
        # regenerate docs/audit/op_target_conformance.md

    python -m tessera.cli.conformance_matrix --check
        # fail with a diff if the on-disk dashboard is stale
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from tessera.compiler import conformance_matrix as cm


_DASHBOARD = Path(__file__).resolve().parents[3] / "docs" / "audit" / "op_target_conformance.md"


def _render() -> str:
    return cm.render_markdown()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--render", action="store_true",
                   help="Regenerate the dashboard.")
    g.add_argument("--check", action="store_true",
                   help="Verify the on-disk dashboard is in sync.")
    ap.add_argument("--out", type=Path, default=_DASHBOARD,
                    help=f"Dashboard path (default: {_DASHBOARD}).")
    args = ap.parse_args(argv)

    rendered = _render()
    if args.render:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"[conformance_matrix] wrote {len(rendered)} bytes → {args.out}")
        counts = cm.status_summary()
        nonzero = {k: v for k, v in counts.items() if v}
        print(f"[conformance_matrix] {nonzero}")
        return 0
    # --check
    if not args.out.is_file():
        print(f"[conformance_matrix] dashboard missing: {args.out}",
              file=sys.stderr)
        return 2
    on_disk = args.out.read_text()
    if on_disk == rendered:
        print(f"[conformance_matrix] in sync ({args.out})")
        return 0
    diff = "".join(difflib.unified_diff(
        on_disk.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=str(args.out) + " (on disk)",
        tofile=str(args.out) + " (regenerated)",
    ))
    print(f"[conformance_matrix] OUT OF SYNC — regen via:"
          f" python -m tessera.cli.conformance_matrix --render",
          file=sys.stderr)
    print(diff, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
