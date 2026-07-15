"""Drift gate for ``tests/README.md`` + ``tests/MEMORY_AND_PERFORMANCE.md``
(Test-tree review phase P1, 2026-05-20).

The reviewer flagged that both docs had bit-rotted (README said
``~4,350`` fast tests when the real number was ~4,748; the
MEMORY_AND_PERFORMANCE table said ``2,214 / 2,990`` when current
collect is ~4,748 / ~5,525).  This gate prevents that from coming
back by re-collecting tests on each CI run and asserting the docs
either carry numbers within a small tolerance or explicitly say
``approximate``.

The gate is **structural** — it checks for plausible numbers, not
exact values.  Tolerance is generous enough to survive ordinary
test-addition churn between doc updates, but tight enough to flag
real staleness (the old doc was off by ~50%).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "tests" / "README.md"
PERF_DOC = ROOT / "tests" / "MEMORY_AND_PERFORMANCE.md"


# Tolerance: any number in the doc must be within ±15% of the
# measured value, OR the doc must explicitly carry "approximate" /
# "~" in front of it.  ±15% absorbs normal week-to-week churn (e.g.,
# a sprint that adds ~500 tests) without forcing a doc update on
# every PR.
_TOLERANCE = 0.15


def _collect_fast_count() -> int:
    """Run ``pytest --collect-only -m 'not slow'`` and parse the count."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests/unit"),
         "--collect-only", "-q", "-m", "not slow"],
        capture_output=True, text=True, check=False,
    )
    # Last non-empty line is the summary line.  Pattern:
    # "4748/5525 tests collected (777 deselected) in 0.51s"
    m = re.search(r"(\d+)/\d+ tests collected", proc.stdout)
    if m is None:
        # Fallback: "5525 tests collected" (no -m filter shape)
        m = re.search(r"(\d+) tests collected", proc.stdout)
    assert m is not None, (
        f"could not parse pytest collect-only output:\n{proc.stdout!r}"
    )
    return int(m.group(1))


def _collect_full_count() -> int:
    """Run ``pytest --collect-only`` (no -m filter) for the full total."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests/unit"),
         "--collect-only", "-q"],
        capture_output=True, text=True, check=False,
    )
    m = re.search(r"(\d+) tests collected", proc.stdout)
    assert m is not None, (
        f"could not parse pytest collect-only output:\n{proc.stdout!r}"
    )
    return int(m.group(1))


def _doc_text_or_skip(path: Path) -> str:
    import pytest

    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} missing — drift gate skipped")
    return path.read_text(encoding="utf-8")


def _within_tolerance(claimed: int, measured: int) -> bool:
    if measured == 0:
        return claimed == 0
    return abs(claimed - measured) / measured <= _TOLERANCE


@pytest.mark.slow
def test_readme_fast_count_is_current() -> None:
    """The README's fast-suite test count must be within ±15% of the
    actual ``pytest --collect-only -m 'not slow'`` result, OR the doc
    must explicitly call its number out as ``approximate`` / ``~``."""
    text = _doc_text_or_skip(README)
    measured = _collect_fast_count()
    # Find numbers preceded by a "fast" hint (~4,750 fast tests).
    # The README's quick-start row uses the pattern ``~N fast tests``
    # or ``~N,NNN tests``.  Look at the first 60 lines.
    head = "\n".join(text.splitlines()[:60])
    # Pattern: "~4,750 fast tests" or "4750 fast tests" or "4,748 fast tests"
    # Require at least one digit so a bare comma doesn't match.
    matches = re.findall(r"~?(\d[\d,]*)\s+(?:fast )?tests?", head)
    assert matches, (
        f"README at {README.relative_to(ROOT)} doesn't mention any "
        f"fast-suite test count — adding ~N text in the Quick start "
        f"section keeps this gate informative.  Measured: {measured}."
    )
    # Accept if any claimed number is within tolerance OR the text
    # carries ``approximate``.
    if "approximate" in head.lower() or "~" in head:
        # Tilde counts as an explicit "approximate" marker.
        for raw in matches:
            claimed = int(raw.replace(",", ""))
            if _within_tolerance(claimed, measured):
                return
        # Even with ~, we still want them within tolerance — the ~
        # is just a hint that the number is approximate.
    # Strict check: at least one claimed number must be in tolerance.
    for raw in matches:
        claimed = int(raw.replace(",", ""))
        if _within_tolerance(claimed, measured):
            return
    raise AssertionError(
        f"README mentions test counts {matches!r} but none are within "
        f"±{int(_TOLERANCE*100)}% of measured fast-suite count "
        f"{measured}.  Update the Quick start table or mark counts "
        f"explicitly as approximate."
    )


@pytest.mark.slow
def test_perf_doc_fast_and_full_counts_are_current() -> None:
    """``MEMORY_AND_PERFORMANCE.md`` must carry fast + full counts
    within ±15% of the live pytest collect numbers."""
    text = _doc_text_or_skip(PERF_DOC)
    fast = _collect_fast_count()
    full = _collect_full_count()
    # Pull all comma-formatted integers from the suite-by-suite
    # table region; the table sits under "## Suite-by-suite footprint".
    table_re = re.search(
        r"## Suite-by-suite footprint([\s\S]*?)(?=^##\s|\Z)",
        text, re.MULTILINE,
    )
    assert table_re, (
        f"{PERF_DOC.relative_to(ROOT)} missing the "
        f"'Suite-by-suite footprint' section header"
    )
    table = table_re.group(1)
    # Numbers like "4,748" or "5,525" or "777".
    nums = [int(s.replace(",", ""))
            for s in re.findall(r"\b([\d,]{3,7})\b", table)]
    # At least one number must be within tolerance of the fast count.
    in_tol_fast = any(_within_tolerance(n, fast) for n in nums)
    in_tol_full = any(_within_tolerance(n, full) for n in nums)
    assert in_tol_fast, (
        f"{PERF_DOC.relative_to(ROOT)} has no number within "
        f"±{int(_TOLERANCE*100)}% of measured fast count {fast}.  "
        f"Numbers found: {nums!r}.  Update the suite-by-suite table."
    )
    assert in_tol_full, (
        f"{PERF_DOC.relative_to(ROOT)} has no number within "
        f"±{int(_TOLERANCE*100)}% of measured full count {full}.  "
        f"Numbers found: {nums!r}.  Update the suite-by-suite table."
    )


def test_readme_does_not_mention_stale_test_failures() -> None:
    """The 'Known pre-existing failures' section had bit-rotted to
    claim ``TestDiffCommand`` was failing when it's actually passing.
    Lock the new (clean) state by requiring the section either say
    'None' or explicitly cite a current failure."""
    text = _doc_text_or_skip(README)
    # Find the 'Known pre-existing failures' section.
    m = re.search(
        r"## Known pre-existing failures\n([\s\S]*?)(?=^##\s|\Z)",
        text, re.MULTILINE,
    )
    if m is None:
        # If the section was removed entirely that's also fine.
        return
    body = m.group(1).strip()
    # Either "None" (case-insensitive at start) or it must NOT cite
    # tests that actually pass today.
    if re.match(r"none\b", body, re.IGNORECASE):
        return
    # If we get here, the section claims a specific failure.  Verify
    # it's not the stale TestDiffCommand claim.
    assert "TestDiffCommand" not in body, (
        "README still cites a 'TestDiffCommand' failure but those "
        "tests pass today.  Either remove the section or replace "
        "it with the current failure (if any)."
    )
