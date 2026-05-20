"""``tessera-gpu-target-map`` — CLI for the NVIDIA + ROCm target maps.

Apple follow-up #3 (2026-05-20).  Parallel to
``tessera-apple-target-map`` but parameterized by ``--target=``.
Today every row is at ``artifact_only`` / ``planned``; the dashboard
will auto-promote rows when Phase G/H hardware bring-up moves the
underlying capability + manifest entries.

Examples:

    python -m tessera.cli.gpu_target_map --target=nvidia_sm90 --render
    python -m tessera.cli.gpu_target_map --target=rocm --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tessera.compiler.gpu_target_map import render_markdown, write_doc


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIT_DIR = _REPO_ROOT / "docs" / "audit" / "generated"

_TARGETS = (
    "nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120",
    "rocm", "rocm_gfx90a", "rocm_gfx940", "rocm_gfx942",
    "rocm_gfx950", "rocm_gfx1100", "rocm_gfx1200",
)


def _doc_path_for(target: str) -> Path:
    return _AUDIT_DIR / f"{target}_target_map.md"


def _cmd_render(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else _doc_path_for(args.target)
    written = write_doc(args.target, out)
    print(f"wrote {written}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    doc_path = _doc_path_for(args.target)
    if not doc_path.exists():
        print(
            f"FAIL: {doc_path} missing — run "
            f"`python -m tessera.cli.gpu_target_map "
            f"--target={args.target} --render` first.",
            file=sys.stderr,
        )
        return 1
    on_disk = doc_path.read_text(encoding="utf-8")
    rendered = render_markdown(args.target)
    if on_disk != rendered:
        print(
            f"FAIL: {doc_path} is out of date.\n"
            f"Run `python -m tessera.cli.gpu_target_map "
            f"--target={args.target} --render` to update.",
            file=sys.stderr,
        )
        return 1
    print(f"[gpu_target_map:{args.target}] clean")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-gpu-target-map",
        description=(
            "Per-op view of NVIDIA / ROCm coverage — same row schema "
            "as ``apple_target_map``."
        ),
    )
    p.add_argument(
        "--target", required=True, choices=_TARGETS,
        help="Target name; see ``capabilities.TARGET_CAPABILITIES``.",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Regenerate the doc.",
    )
    p.add_argument(
        "--check", action="store_true",
        help="CI gate: fail if the on-disk doc is stale.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path override.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.render:
        return _cmd_render(args)
    if args.check:
        return _cmd_check(args)
    p.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
