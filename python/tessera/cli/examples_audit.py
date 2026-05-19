"""``tessera-examples-audit`` — execute manifest entry points and report.

The audit is the runtime half of the examples surface gate.  It pairs
with ``tessera.cli.claim_lint`` which scans README language; this
module actually runs the scripts.

Subcommands
-----------

``--check``  (default)
    Execute every ``runnable`` entry and every ``runnable_optional``
    entry whose ``extras_required`` are importable.  Non-zero exit
    means at least one entry failed or the manifest filesystem audit
    raised an issue.

``--render``
    Write ``docs/audit/generated/examples_status.md`` from the
    manifest.  Used to refresh the doc after manifest edits.

``--list``
    Print the manifest as a table.  No execution.

The CLI is registered as a console script via ``pyproject.toml``;
``python -m tessera.cli.examples_audit`` is the equivalent module form.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from tessera.compiler.examples_manifest import (
    ExampleEntry,
    all_entries,
    audit_filesystem,
    render_markdown,
    status_counts,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DOC = _REPO_ROOT / "docs" / "audit" / "generated" / "examples_status.md"


def _ansi(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


# Match the bare ``python`` token only when it appears at the start
# of the command (a fresh argv[0] position) — i.e., either at
# string-start or directly preceded by whitespace, AND followed by
# whitespace or end.  This intentionally does NOT rewrite occurrences
# inside ``PYTHONPATH=...`` (where ``python`` is a directory name).
_PYTHON_TOKEN_RE = re.compile(r"(?:^|(?<=\s))python(?=\s|$)")


def _resolve_command(cmd: str) -> str:
    """Substitute the bare ``python`` token with ``sys.executable``.

    CI environments often expose only ``python3`` (or the venv's
    interpreter) on ``PATH``; the manifest commands write ``python``
    for human readability and we rewrite at exec time.

    Only argv[0]-position tokens are rewritten; directory names like
    ``PYTHONPATH=python`` are left alone.
    """

    return _PYTHON_TOKEN_RE.sub(sys.executable, cmd)


# Matches a leading ``VAR=value`` token (no spaces in the value, since
# the manifest uses ``PYTHONPATH=a:b python ...`` form).
_ENV_PREFIX_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(\S+)$")


def _split_env_prefix(cmd: str) -> tuple[dict[str, str], list[str]]:
    """Peel ``VAR=value`` prefixes off the front of a command.

    Returns ``(env_overrides, argv)``.  ``argv`` is suitable for
    ``subprocess.run(shell=False)``.
    """

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


def _run_entry(entry: ExampleEntry, *, timeout: int) -> tuple[bool, str]:
    """Execute one entry's command. Returns (passed, captured_tail)."""

    if not entry.command:
        return True, "(no command — skipped)"
    cmd = _resolve_command(entry.command)
    # Commands may carry ``VAR=value`` env-var prefixes (the manifest
    # uses ``PYTHONPATH=python python ...``); pull those off and pass
    # them via ``env=`` so we can avoid ``shell=True`` and the
    # associated injection-vector linter warning.  Manifest commands
    # are trusted developer-curated strings, but spawning without a
    # shell is still preferable.
    env_overrides, argv = _split_env_prefix(cmd)
    try:
        proc = subprocess.run(
            argv,
            shell=False,  # noqa: S603 — argv is built from a trusted manifest
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                **env_overrides,
                # Force unbuffered output so partial logs still
                # show up if a run hangs.
                "PYTHONUNBUFFERED": "1",
            },
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    ok = proc.returncode == 0
    tail = (proc.stdout or "")[-500:].strip()
    if not ok:
        tail += "\n--- stderr ---\n" + (proc.stderr or "")[-500:].strip()
    return ok, tail


def _cmd_check(args: argparse.Namespace) -> int:
    issues = audit_filesystem()
    if issues:
        print(_ansi("31", "FAIL — manifest filesystem audit:"))
        for issue in issues:
            print(f"  - {issue}")
        return 2

    entries = all_entries()
    by_status = status_counts()
    print(f"[examples_audit] {len(entries)} entries  {dict(by_status)}")

    failures: list[tuple[ExampleEntry, str]] = []
    skipped: list[tuple[ExampleEntry, str]] = []
    passed: int = 0

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
                    if not entry.resolve_extras_available()
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
        else:  # scaffold / broken
            print(
                f"  {_ansi('90', 'NOT-RUN')}  {entry.directory}  "
                f"({entry.status})"
            )

    print(
        f"\n[examples_audit] passed={passed} "
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
    out_path = Path(args.output) if args.output else _GENERATED_DOC
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    by_status = status_counts()
    width = max(len(e.directory) for e in all_entries())
    print(f"{'directory':<{width}}  status            entry_point")
    for entry in all_entries():
        print(
            f"{entry.directory:<{width}}  "
            f"{entry.status:<17} {entry.entry_point}"
        )
    print()
    print(f"counts: {dict(by_status)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-examples-audit",
        description=(
            "Execute and audit every active example entry point declared "
            "in tessera.compiler.examples_manifest."
        ),
    )
    sub = p.add_subparsers(dest="cmd")
    p_check = sub.add_parser(
        "check", help="Run runnable entries and report failures."
    )
    p_check.add_argument(
        "--timeout", type=int, default=180,
        help="Per-entry timeout in seconds (default: 180).",
    )
    p_render = sub.add_parser(
        "render", help="Regenerate docs/audit/generated/examples_status.md."
    )
    p_render.add_argument(
        "--output", default=None,
        help="Output path (default: docs/audit/generated/examples_status.md)",
    )
    sub.add_parser("list", help="Print the manifest table.")
    # Backwards-friendly aliases so '--check / --render / --list' work
    # without a subcommand verb.
    p.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--render", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--timeout", type=int, default=180, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--output", default=None, help=argparse.SUPPRESS
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "check" or args.check:
        return _cmd_check(args)
    if args.cmd == "render" or args.render:
        return _cmd_render(args)
    if args.cmd == "list" or args.list:
        return _cmd_list(args)
    # Default: --check
    return _cmd_check(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
