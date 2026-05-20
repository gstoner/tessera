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


SURFACES = ("examples", "benchmarks", "research", "tools", "tests")


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


# Sentinel SurfaceEntry used to fill the ``entry`` field on
# api_reference violations — they aren't tied to a real manifest
# row but ClaimViolation requires the field for uniform display.
_API_REF_SENTINEL_ENTRY = SurfaceEntry(
    directory="docs/reference",
    entry_point="docs/reference/tessera-api-reference.md",
    status="archived",  # not a runnable surface; just a doc target
    reason="api_reference doc sentinel for claim_lint",
)


def find_api_reference_op_violations() -> list[ClaimViolation]:
    """Phase P1-8 (Test-tree review, 2026-05-20).

    Scan ``docs/reference/tessera-api-reference.md`` for any
    ``tessera.<module>.<symbol>`` reference (where ``<module>`` is one
    of ``ops`` / ``compiler`` / ``nn`` / ``autodiff`` / ``losses`` /
    ``optim``) and assert each ``<symbol>`` is reachable on the live
    ``tessera`` package.  This catches API drift in the doc — e.g.,
    a renamed compiler helper or a removed op that the doc still
    references.

    Returns an empty list when the doc is clean.  Each violation
    carries the file + line + the unresolved symbol.
    """
    doc = _REPO_ROOT / "docs" / "reference" / "tessera-api-reference.md"
    if not doc.exists():
        return []
    import tessera

    out: list[ClaimViolation] = []
    # Match ``tessera.<module>.<symbol>`` inside backticks or prose.
    # We only target known sub-modules to keep noise low; ad-hoc
    # references to e.g. ``tessera.runtime.<X>`` aren't part of the
    # public-surface contract.
    pat = re.compile(
        r"tessera\.(ops|compiler|nn|autodiff|losses|optim)"
        r"\.([a-zA-Z_][a-zA-Z0-9_]*)"
    )
    for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        for m in pat.finditer(line):
            module, symbol = m.group(1), m.group(2)
            try:
                mod = getattr(tessera, module)
            except AttributeError:
                # ``tessera.<module>`` itself missing — bigger problem,
                # report it once.
                out.append(ClaimViolation(
                    surface="api_reference",
                    entry=_API_REF_SENTINEL_ENTRY,
                    file=doc,
                    line_no=lineno,
                    line=line.strip(),
                    pattern=f"tessera.{module}",
                    description=(
                        f"unresolved module ``tessera.{module}`` "
                        "referenced in API ref"
                    ),
                ))
                continue
            # Resolve as attribute first; if that fails, try as a
            # submodule (``tessera.compiler.clifford_jit`` is a
            # submodule that ``getattr(tessera.compiler, 'clifford_jit')``
            # won't find unless the parent has imported it eagerly).
            resolved = hasattr(mod, symbol)
            if not resolved:
                import importlib
                try:
                    importlib.import_module(f"tessera.{module}.{symbol}")
                    resolved = True
                except ImportError:
                    resolved = False
            if not resolved:
                out.append(ClaimViolation(
                    surface="api_reference",
                    entry=_API_REF_SENTINEL_ENTRY,
                    file=doc,
                    line_no=lineno,
                    line=line.strip(),
                    pattern=f"tessera.{module}.{symbol}",
                    description=(
                        f"``tessera.{module}.{symbol}`` is referenced "
                        "in the API ref but does not exist on the "
                        "live package"
                    ),
                ))
    return out


def _cmd_check(args: argparse.Namespace) -> int:
    violations = find_violations(args.surface)
    # Phase P1-8: also scan the API reference doc for op references
    # that no longer resolve.  This catches the doc-side of API drift
    # (the surface-side is covered by find_violations above).
    if args.surface in (None, "api_reference"):
        violations.extend(find_api_reference_op_violations())
    scope = args.surface or "all surfaces + api_reference"
    if not violations:
        print(
            f"[claim_lint:{scope}] clean — "
            f"{_scanned_count(args.surface)} "
            f"scaffold/broken/archived directories scanned + "
            f"API reference op-name references resolved."
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
