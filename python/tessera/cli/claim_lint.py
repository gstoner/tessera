"""``tessera-claim-lint`` — flag README overclaim language vs. manifest status.

Scans the README of every ``scaffold`` / ``broken`` / ``archived``
directory across the four audited surfaces (``examples``,
``benchmarks``, ``research``, ``tools``) for language that implies
the row is runnable today — phrases like ``runnable``, ``run this``,
``working demo``, ``end-to-end``, ``CPU-only smoke``,
``self-contained``.  Any such phrase is reported as a violation.

The lint is intentionally narrow:

* It only walks directories declared in the relevant per-surface
  manifest — README files outside the manifests are ignored.
* It only fires on ``scaffold`` / ``broken`` / ``archived`` rows.
  ``runnable`` / ``runnable_optional`` / ``compile_only`` rows are
  allowed to use whatever language they like since the executable
  audit covers them.
* Phrases inside ``<!-- ... -->`` HTML comments and fenced code
  blocks are ignored — the lint targets prose, not snippets that
  illustrate what *should* happen once the scaffold is finished.
* A line containing ``STATUS:`` (case-insensitive) is treated as an
  explicit status disclaimer and skipped.

Run via:

    python -m tessera.cli.claim_lint --check                     # all surfaces
    python -m tessera.cli.claim_lint --surface=examples --check  # one surface

Exit code 0 means clean; 1 means at least one violation.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tessera.compiler.surface_manifest import SurfaceEntry


_REPO_ROOT = Path(__file__).resolve().parents[3]


SURFACES = ("examples", "benchmarks", "research", "tools")


# Statuses that trigger the lint.  ``archived`` rows are scanned too
# because they're easy to forget — an archived sample README that
# still says "runnable" is a stale claim.
_LINTED_STATUSES = ("scaffold", "broken", "archived")


# Phrases that imply the entry actually runs on CI.  Patterns are
# matched word-boundary-aware and case-insensitive.
_OVERCLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bruns?\s+(?:on\s+)?(?:cpu(?:-only)?|in\s+ci)\b", "claims a CPU/CI run"),
    (r"\bworking\s+(?:demo|example|implementation)\b", "claims to be a 'working' demo"),
    (r"\bend[-\s]?to[-\s]?end\s+(?:demo|run|example)\b", "claims an end-to-end run"),
    (r"\bCI[-\s]?runnable\b", "claims to be 'CI-runnable'"),
    (r"\bproduction[-\s]?ready\b", "claims to be 'production-ready'"),
    (r"\bself[-\s]?contained\s+(?:demo|example|run)\b", "claims to be 'self-contained'"),
    (r"\bship[s]?\s+(?:a\s+)?runnable\b", "claims to ship a runnable demo"),
)

_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ClaimViolation:
    surface: str
    entry: SurfaceEntry
    file: Path
    line_no: int
    line: str
    pattern: str
    description: str


def _strip_codeblocks_and_comments(text: str) -> str:
    cleaned = _FENCED_BLOCK.sub("", text)
    cleaned = _HTML_COMMENT.sub("", cleaned)
    return cleaned


def _scan_file(
    surface: str, entry: SurfaceEntry, path: Path
) -> list[ClaimViolation]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    cleaned = _strip_codeblocks_and_comments(raw)
    violations: list[ClaimViolation] = []
    for line_no, line in enumerate(cleaned.splitlines(), start=1):
        if "status:" in line.lower():
            continue
        for pattern, description in _OVERCLAIM_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                violations.append(
                    ClaimViolation(
                        surface=surface,
                        entry=entry,
                        file=path,
                        line_no=line_no,
                        line=line.strip(),
                        pattern=pattern,
                        description=description,
                    )
                )
    return violations


def _manifest_module(surface: str):
    return importlib.import_module(f"tessera.compiler.{surface}_manifest")


def find_violations(
    surface: str | None = None,
) -> list[ClaimViolation]:
    """Return every claim-lint violation across the configured surfaces.

    ``surface=None`` walks all four surfaces; otherwise restricts to
    one.
    """

    surfaces_to_walk = (surface,) if surface else SURFACES
    out: list[ClaimViolation] = []
    for s in surfaces_to_walk:
        mod = _manifest_module(s)
        rows: list[SurfaceEntry] = []
        for status in _LINTED_STATUSES:
            rows.extend(mod.entries_by_status(status))
        for entry in rows:
            d = entry.directory_path
            for candidate in ("README.md", "Readme.md", "readme.md"):
                path = d / candidate
                if path.exists():
                    out.extend(_scan_file(s, entry, path))
                    break
    return out


def _scanned_count(surface: str | None) -> int:
    surfaces_to_walk = (surface,) if surface else SURFACES
    total = 0
    for s in surfaces_to_walk:
        mod = _manifest_module(s)
        for status in _LINTED_STATUSES:
            total += len(mod.entries_by_status(status))
    return total


def _cmd_check(args: argparse.Namespace) -> int:
    violations = find_violations(args.surface)
    scope = args.surface or "all surfaces"
    if not violations:
        print(
            f"[claim_lint:{scope}] clean — "
            f"{_scanned_count(args.surface)} "
            f"scaffold/broken/archived directories scanned."
        )
        return 0
    print(f"[claim_lint:{scope}] FAIL — {len(violations)} overclaim(s):")
    for v in violations:
        rel = v.file.relative_to(_REPO_ROOT)
        print(
            f"  [{v.surface}] {rel}:{v.line_no}  "
            f"({v.entry.status})  {v.description}"
        )
        print(f"    > {v.line}")
    print()
    print(
        "Fix by either (a) downgrading the language to match the "
        "scaffold/broken/archived status declared in the surface's "
        "manifest, or (b) promoting the manifest row to runnable / "
        "runnable_optional / compile_only and adding a passing "
        "executable audit row."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-claim-lint",
        description=(
            "Flag README overclaim language on scaffold/broken/"
            "archived rows across the four audited surfaces."
        ),
    )
    p.add_argument(
        "--surface",
        choices=list(SURFACES),
        default=None,
        help=(
            "Restrict the lint to a single surface.  Defaults to all "
            "four surfaces."
        ),
    )
    p.add_argument(
        "--check", action="store_true", default=True,
        help="Run the lint and report violations (default).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return _cmd_check(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
