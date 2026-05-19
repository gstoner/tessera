"""``tessera-claim-lint`` — flag README overclaim language vs. manifest status.

Scans every active example directory's ``README.md`` (and ``STATUS.md``
when present) for language that implies the example is runnable —
phrases like ``runnable``, ``run this``, ``working demo``,
``end-to-end``, ``CPU-only smoke``, ``self-contained``.  Any such
phrase on a row whose manifest status is ``scaffold`` or ``broken``
is reported as a claim_lint violation.

The lint is intentionally narrow:

* It only walks directories declared in
  ``tessera.compiler.examples_manifest`` — README files outside the
  manifest are ignored.
* It only fires on ``scaffold`` / ``broken`` rows.  ``runnable``,
  ``runnable_optional``, and ``compile_only`` rows are allowed to use
  whatever language they like since the executable audit covers them.
* Phrases inside ``<!-- ... -->`` HTML comments and fenced code blocks
  are ignored — the lint targets prose claims, not snippets that
  illustrate what *should* happen once the scaffold is finished.
* A line containing ``STATUS:`` (case-insensitive) is treated as an
  explicit status disclaimer and skipped.

Run via:

    python -m tessera.cli.claim_lint --check

Exit code 0 means clean; 1 means at least one violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tessera.compiler.examples_manifest import (
    ExampleEntry,
    entries_by_status,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


# Phrases that imply the example actually runs on CI.  Patterns are
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
    entry: ExampleEntry
    file: Path
    line_no: int
    line: str
    pattern: str
    description: str


def _strip_codeblocks_and_comments(text: str) -> str:
    """Strip fenced code blocks + HTML comments before pattern scan."""

    cleaned = _FENCED_BLOCK.sub("", text)
    cleaned = _HTML_COMMENT.sub("", cleaned)
    return cleaned


def _scan_file(entry: ExampleEntry, path: Path) -> list[ClaimViolation]:
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
            # Explicit status disclaimer line — skip.
            continue
        for pattern, description in _OVERCLAIM_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                violations.append(
                    ClaimViolation(
                        entry=entry,
                        file=path,
                        line_no=line_no,
                        line=line.strip(),
                        pattern=pattern,
                        description=description,
                    )
                )
    return violations


def find_violations(
    entries: Iterable[ExampleEntry] | None = None,
) -> list[ClaimViolation]:
    """Return every claim-lint violation across the configured manifest."""

    rows = tuple(entries) if entries is not None else entries_by_status("scaffold")
    # Also include 'broken' rows when callers don't restrict the set.
    if entries is None:
        rows = rows + entries_by_status("broken")
    out: list[ClaimViolation] = []
    for entry in rows:
        d = entry.directory_path
        for candidate in ("README.md", "Readme.md", "readme.md"):
            path = d / candidate
            if path.exists():
                out.extend(_scan_file(entry, path))
                break
    return out


def _cmd_check(args: argparse.Namespace) -> int:
    violations = find_violations()
    if not violations:
        print(
            f"[claim_lint] clean — "
            f"{len(entries_by_status('scaffold')) + len(entries_by_status('broken'))} "
            f"scaffold/broken directories scanned."
        )
        return 0
    print(f"[claim_lint] FAIL — {len(violations)} overclaim(s):")
    for v in violations:
        rel = v.file.relative_to(_REPO_ROOT)
        print(f"  {rel}:{v.line_no}  ({v.entry.status})  {v.description}")
        print(f"    > {v.line}")
    print()
    print(
        "Fix by either (a) downgrading the language to match the "
        "scaffold/broken status declared in "
        "python/tessera/compiler/examples_manifest.py, or "
        "(b) promoting the manifest row to runnable / "
        "runnable_optional / compile_only and adding a passing "
        "executable audit row."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessera-claim-lint",
        description=(
            "Flag README overclaim language on scaffold/broken example "
            "directories."
        ),
    )
    p.add_argument("--check", action="store_true", default=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return _cmd_check(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
