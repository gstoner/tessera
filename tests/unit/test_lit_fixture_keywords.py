"""Lit directive keywords must not appear outside a directive position.

lit's integrated-test parser matches its keywords with an UNANCHORED regex
over every line of a fixture (`(RUN:|XFAIL:|REQUIRES:|UNSUPPORTED:|...)(.*)`),
so the token does not have to start a comment to be obeyed. Two ways that
bites, both silent about their cause:

* a `CHECK:` line naming a diagnostic such as `..._UNSUPPORTED:` makes lit
  parse the rest of the line as an UNSUPPORTED boolean expression and mark
  the test UNRESOLVED (hit 2026-09-15 in `apple_matmul2d_lowering_invalid`);
* an uppercase `END.` anywhere in prose (`// ... the NVIDIA BACKEND.`) ends
  script parsing, so RUN lines after it are never executed and the test
  passes having run nothing.

This gate scans every lit suite in the tree. A keyword is allowed only as a
real directive: the first token after the comment leader. Inside a FileCheck
pattern, split the token with a regex brace (`{{UNSUPPORTED}}:`), which lit
does not recognize and FileCheck matches literally.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# lit's own list (lit.TestRunner._parseKeywords), case-sensitive.
KEYWORDS = ("RUN:", "XFAIL:", "REQUIRES:", "UNSUPPORTED:", "ALLOW_RETRIES:", "END.", "DEFINE:", "REDEFINE:")
_DIRECTIVE = re.compile(r"^\s*(?://|#|;)\s*(RUN|XFAIL|REQUIRES|UNSUPPORTED|ALLOW_RETRIES|DEFINE|REDEFINE):|^\s*(?://|#|;)\s*END\.\s*$")


def _lit_suites() -> list[Path]:
    return sorted(p.parent for p in ROOT.rglob("lit.cfg.py")
                  if "build" not in p.parts and "archive" not in p.parts and ".git" not in p.parts)


def _fixtures() -> list[Path]:
    out: list[Path] = []
    for suite in _lit_suites():
        out += sorted(f for f in suite.rglob("*.mlir") if "archive" not in f.parts)
    return out


def test_lit_suites_are_discovered():
    assert _lit_suites(), "no lit.cfg.py found; the keyword gate has nothing to protect"


def test_lit_keywords_only_appear_as_directives():
    offenders: list[str] = []
    for fixture in _fixtures():
        for number, line in enumerate(fixture.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            hits = [k for k in KEYWORDS if k in line]
            if not hits or _DIRECTIVE.match(line):
                continue
            offenders.append(f"{fixture.relative_to(ROOT)}:{number}: {hits} in a non-directive line "
                             f"(lit obeys it anyway; split it as `{{{{{hits[0].rstrip(':.')}}}}}`): {line.strip()[:100]}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("line,ok", [
    ("// RUN: tessera-opt %s | FileCheck %s", True),
    ("// CHECK: APPLE_X_{{UNSUPPORTED}}: operand a", True),
    ("// CHECK: APPLE_X_UNSUPPORTED: operand a", False),
    ("// the NVIDIA BACKEND. Then the next RUN line is dropped", False),
    ("// END.", True),
    ("//   REQUIRES: tessera-apple-backend", True),
])
def test_directive_classifier(line, ok):
    hits = [k for k in KEYWORDS if k in line]
    assert (not hits or bool(_DIRECTIVE.match(line))) is ok


def test_every_discovered_fixture_has_a_run_line():
    """A `.mlir` inside a lit suite with no `RUN:` line is Unresolved, and one
    Unresolved test fails the whole suite.

    Data files driven by a Python test are the recurring cause, and the marker
    that looks like it should suppress them does not: on LLVM 23's lit a test
    with no RUN line is Unresolved *before* its `UNSUPPORTED:` is consulted. The
    remedy is to keep such a file outside lit discovery — `tests/fixtures/` is
    where the x86 and ROCm composed-layout inputs live for exactly this reason.

    The ROCm instance sat on main behind an `UNSUPPORTED: true` marker, failing
    `check-tessera-rocm`, which is that backend's only automated fixture coverage
    and which no PR check runs.
    """
    missing = [f.relative_to(ROOT) for f in _fixtures()
               if not any(_DIRECTIVE.match(line) and "RUN" in line.split(":")[0]
                          for line in f.read_text(encoding="utf-8", errors="replace").splitlines())]
    assert not missing, (
        "lit fixtures with no RUN: line (each one makes its whole suite fail as "
        f"Unresolved). Move Python-driven data to tests/fixtures/: {missing}")
