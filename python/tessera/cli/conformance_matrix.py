"""CLI entry point for the op×target conformance matrix dashboard.

Usage::

    python -m tessera.cli.conformance_matrix --render
        # regenerate docs/audit/op_target_conformance.md

    python -m tessera.cli.conformance_matrix --check
        # fail with a diff if the on-disk dashboard is stale

    python -m tessera.cli.conformance_matrix --verify-fixtures
        # invoke pytest on every (op, target) fixture declared in the
        # manifest's _NUMERICAL_FIXTURES map; exit non-zero on any
        # failure. Replaces the dashboard's "numerical_check is ✅"
        # claim with actual proof.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tessera.compiler import backend_manifest as bm
from tessera.compiler import conformance_matrix as cm


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHBOARD = _REPO_ROOT / "docs" / "audit" / "op_target_conformance.md"


def _render() -> str:
    return cm.render_markdown()


@dataclass(frozen=True)
class FixtureResult:
    op: str
    target: str
    fixture: str
    ok: bool
    elapsed_s: float
    short_message: str  # last non-empty line of stdout, truncated


def _verify_fixtures(*, pytest_args: tuple[str, ...] = ()) -> int:
    """Invoke pytest on every declared ``execute_compare_fixture``,
    print a per-cell pass/fail summary, and exit non-zero on any
    failure. The dashboard's ``numerical_check`` column reports ``✅``
    only when the manifest declares a fixture; this verifier turns
    that declaration into actual proof.

    Files are de-duplicated — a single test file may cover multiple
    (op, target) pairs; we run it once and report the same result
    against every pair that points at it.
    """
    fixtures: dict[str, list[tuple[str, str]]] = {}
    for (op, target), rel in bm._NUMERICAL_FIXTURES.items():
        fixtures.setdefault(rel, []).append((op, target))

    print(f"[verify-fixtures] {len(bm._NUMERICAL_FIXTURES)} declared "
          f"(op, target) pair(s) across {len(fixtures)} file(s)")
    print()

    results: list[FixtureResult] = []
    for rel, pairs in sorted(fixtures.items()):
        path = _REPO_ROOT / rel
        if not path.is_file():
            print(f"[verify-fixtures] FAIL {rel}  fixture file missing  "
                  f"covers: {', '.join(f'{op}/{t}' for op, t in pairs)}")
            for (op, target) in pairs:
                results.append(FixtureResult(
                    op=op, target=target, fixture=rel, ok=False,
                    elapsed_s=0.0,
                    short_message=f"fixture file missing: {rel}",
                ))
            continue

        cmd = [sys.executable, "-m", "pytest", str(path), "-q",
               "--no-header", *pytest_args]
        env_repo = str(_REPO_ROOT / "python")
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=str(_REPO_ROOT),
                env={**__import__("os").environ, "PYTHONPATH": env_repo},
                capture_output=True, text=True, timeout=300,
            )
            elapsed = time.perf_counter() - start
            ok = proc.returncode == 0
            # Extract the last non-empty stdout line as a short message.
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            short = lines[-1] if lines else (
                proc.stderr.splitlines()[-1] if proc.stderr.splitlines()
                else "(no output)")
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            ok = False
            short = "pytest timed out after 300s"

        sym = "PASS" if ok else "FAIL"
        print(f"[verify-fixtures] {sym:4} {rel}  ({elapsed:.1f}s)  "
              f"covers: {', '.join(f'{op}/{t}' for op, t in pairs)}")
        if not ok:
            print(f"               └─ {short[:200]}")
        for (op, target) in pairs:
            results.append(FixtureResult(
                op=op, target=target, fixture=rel, ok=ok,
                elapsed_s=elapsed, short_message=short[:200],
            ))

    n_pass = sum(1 for r in results if r.ok)
    n_fail = sum(1 for r in results if not r.ok)
    print()
    print(f"[verify-fixtures] summary: "
          f"{n_pass}/{len(results)} (op, target) pair(s) PASS, "
          f"{n_fail} FAIL")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--render", action="store_true",
                   help="Regenerate the dashboard.")
    g.add_argument("--check", action="store_true",
                   help="Verify the on-disk dashboard is in sync.")
    g.add_argument("--verify-fixtures", action="store_true",
                   help="Run pytest on every declared execute_compare_fixture; "
                        "exit non-zero on any failure.")
    ap.add_argument("--out", type=Path, default=_DASHBOARD,
                    help=f"Dashboard path (default: {_DASHBOARD}).")
    ap.add_argument("--pytest-arg", action="append", default=[],
                    help="Extra arg to pass to pytest (repeatable).")
    args = ap.parse_args(argv)

    if args.verify_fixtures:
        return _verify_fixtures(pytest_args=tuple(args.pytest_arg))

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
    print("[conformance_matrix] OUT OF SYNC — regen via:"
          " python -m tessera.cli.conformance_matrix --render",
          file=sys.stderr)
    print(diff, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
