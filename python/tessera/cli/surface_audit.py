"""``tessera-surface-audit`` — unified audit driver for all four surfaces.

Single CLI for the per-surface manifests:

* ``examples``     — backed by ``tessera.compiler.examples_manifest``
* ``benchmarks``   — backed by ``tessera.compiler.benchmarks_manifest``
* ``research``     — backed by ``tessera.compiler.research_manifest``
* ``tools``        — backed by ``tessera.compiler.tools_manifest``
* ``tests``        — backed by ``tessera.compiler.tests_manifest``

Subcommands
-----------

``--check``  (default)
    Execute every ``runnable`` row and every ``runnable_optional``
    row whose ``extras_required`` are importable.  Non-zero exit
    means at least one entry failed or the manifest filesystem audit
    raised an issue.

``--render``
    Write ``docs/audit/generated/<surface>_status.md`` from the
    manifest.

``--list``
    Print the manifest table.  No execution.

``--surface=<name>``
    Required.  Selects which manifest to operate on.

The legacy ``tessera-examples-audit`` console script remains as a
thin wrapper that defaults ``--surface=examples``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from tessera.compiler.surface_manifest import SurfaceEntry


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIT_DIR = _REPO_ROOT / "docs" / "audit" / "generated"

SURFACES = ("examples", "benchmarks", "research", "tools", "tests")


def _manifest_module(surface: str):
    mod_name = f"tessera.compiler.{surface}_manifest"
    return importlib.import_module(mod_name)


def _generated_doc_path(surface: str) -> Path:
    return _AUDIT_DIR / f"{surface}_status.md"


# Match the bare ``python`` token only when it appears at the start
# of the command (a fresh argv[0] position).  This intentionally does
# NOT rewrite occurrences inside ``PYTHONPATH=...``.
_PYTHON_TOKEN_RE = re.compile(r"(?:^|(?<=\s))python(?=\s|$)")


def _resolve_command(cmd: str) -> str:
    """Substitute the bare ``python`` token with ``sys.executable``."""

    return _PYTHON_TOKEN_RE.sub(sys.executable, cmd)


# Matches a leading ``VAR=value`` token (no spaces in the value).
_ENV_PREFIX_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(\S+)$")


def _split_env_prefix(cmd: str) -> tuple[dict[str, str], list[str]]:
    """Peel ``VAR=value`` prefixes off the front of a command."""

    import shlex

    tokens = shlex.split(cmd)
    env_overrides: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        match = _ENV_PREFIX_RE.match(tokens[i])
        if match is None:
            break
        env_overrides[match.group(1)] = match.group(2)
        i += 1
    return env_overrides, tokens[i:]


def _ansi(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _run_entry(entry: SurfaceEntry, *, timeout: int) -> tuple[bool, str]:
    """Execute one entry's command. Returns (passed, captured_tail).

    Supports ``&&``-chained commands by splitting at the boundary
    and running each step in sequence (still shell=False — each
    step is itself an argv-style invocation).  Per-step env-var
    prefixes (``PYTHONPATH=python python ...``) are honored.
    """

    if not entry.command:
        return True, "(no command — skipped)"
    cmd = _resolve_command(entry.command)
    # Split on `` && `` token; preserves quoted args inside each step.
    steps = [s.strip() for s in cmd.split("&&")]
    last_tail = ""
    for step in steps:
        env_overrides, argv = _split_env_prefix(step)
        if not argv:
            continue
        try:
            proc = subprocess.run(
                argv,
                shell=False,  # noqa: S603 — argv from trusted manifest
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    **env_overrides,
                    "PYTHONUNBUFFERED": "1",
                },
            )
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT after {timeout}s on step: {step}"
        last_tail = (proc.stdout or "")[-500:].strip()
        if proc.returncode != 0:
            last_tail += (
                f"\n--- stderr (step: {step}) ---\n"
                + (proc.stderr or "")[-500:].strip()
            )
            return False, last_tail
    return True, last_tail


def _cmd_check(args: argparse.Namespace) -> int:
    surface = args.surface
    mod = _manifest_module(surface)
    issues = mod.audit_filesystem()
    if issues:
        print(_ansi("31", f"FAIL — {surface} manifest filesystem audit:"))
        for issue in issues:
            print(f"  - {issue}")
        return 2

    entries = mod.all_entries()
    by_status = mod.status_counts()
    print(
        f"[surface_audit:{surface}] {len(entries)} entries  "
        f"{dict(by_status)}"
    )

    failures: list[tuple[SurfaceEntry, str]] = []
    skipped: list[tuple[SurfaceEntry, str]] = []
    passed = 0

    for entry in entries:
        if entry.status == "runnable":
            ok, tail = _run_entry(entry, timeout=args.timeout)
            tag = _ansi("32", "PASS") if ok else _ansi("31", "FAIL")
            print(f"  {tag}  {entry.directory}  ({entry.status})")
            if ok:
                passed += 1
            else:
                failures.append((entry, tail))
        elif entry.status == "runnable_optional":
            if not entry.resolve_extras_available():
                missing = ", ".join(
                    m for m in entry.extras_required
                    if importlib.util.find_spec(m) is None
                )
                print(
                    f"  {_ansi('33', 'SKIP')}  {entry.directory}  "
                    f"({entry.status} — extras missing: {missing})"
                )
                skipped.append((entry, f"missing extras: {missing}"))
                continue
            ok, tail = _run_entry(entry, timeout=args.timeout)
            tag = _ansi("32", "PASS") if ok else _ansi("31", "FAIL")
            print(f"  {tag}  {entry.directory}  ({entry.status})")
            if ok:
                passed += 1
            else:
                failures.append((entry, tail))
        elif entry.status == "compile_only":
            ok, tail = _run_entry(entry, timeout=args.timeout)
            tag = _ansi("32", "PASS") if ok else _ansi("31", "FAIL")
            print(f"  {tag}  {entry.directory}  ({entry.status})")
            if ok:
                passed += 1
            else:
                failures.append((entry, tail))
        else:  # scaffold / broken / archived
            print(
                f"  {_ansi('90', 'NOT-RUN')}  {entry.directory}  "
                f"({entry.status})"
            )

    print(
        f"\n[surface_audit:{surface}] passed={passed} "
        f"failed={len(failures)} skipped={len(skipped)}"
    )
    if failures:
        for entry, tail in failures:
            print()
            print(_ansi("31", f"FAIL  {entry.directory}"))
            print(f"  command: {entry.command}")
            for line in tail.splitlines():
                print(f"  | {line}")
        return 1
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    # The per-surface ``*_status.md`` docs were consolidated (2026-06-04)
    # into the single ``surface_status`` dashboard owned by the
    # generated-doc registry. ``--render`` now regenerates that one
    # consolidated doc (CSV + Markdown) so it cannot recreate orphan
    # per-surface files. ``--surface`` still selects which manifest the
    # ``--check`` smoke gate runs.
    from tessera.compiler import generated_docs as gd

    for p in gd.write(gd.get("surface_status")):
        print(f"wrote {p}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    surface = args.surface
    mod = _manifest_module(surface)
    entries = mod.all_entries()
    if not entries:
        print(f"({surface} manifest is empty)")
        return 0
    width = max(len(e.directory) for e in entries)
    print(f"{'directory':<{width}}  status              entry_point")
    for entry in entries:
        print(
            f"{entry.directory:<{width}}  "
            f"{entry.status:<19} {entry.entry_point}"
        )
    print()
    print(f"counts: {dict(mod.status_counts())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-surface-audit",
        description=(
            "Unified audit driver for the four per-surface manifests "
            "(examples / benchmarks / research / tools / tests)."
        ),
    )
    p.add_argument(
        "--surface",
        required=True,
        choices=list(SURFACES),
        help="Which manifest to operate on.",
    )
    p.add_argument(
        "--check", action="store_true",
        help="Execute runnable rows and report failures (default).",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Regenerate the surface's status doc.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the manifest table.",
    )
    p.add_argument(
        "--timeout", type=int, default=300,
        help="Per-entry timeout in seconds (default: 300).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path for --render (default: derived from surface).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.render:
        return _cmd_render(args)
    if args.list:
        return _cmd_list(args)
    return _cmd_check(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
