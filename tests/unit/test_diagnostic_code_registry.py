"""Arch-1 (2026-05-22) — diagnostic code registry drift gate.

The registry at ``python/tessera/compiler/diagnostic_codes.py`` is the
single Python-side source of truth for MLIR verifier / pass diagnostic
codes that Tessera's C++ layer emits (codes like
``SYMDIM_BINDING_VIOLATION``, ``QUEUE_PUSH_QUEUE_PROVENANCE``).

This is a separate registry from the JIT-level ``JitDiagnosticCode`` /
``FallbackReason`` taxonomy at ``tests/unit/test_diagnostic_codes.py``
(P0-2 — that surface lives in ``python/tessera/compiler/jit/``).  The
two registries don't overlap:

  * ``JitDiagnosticCode`` covers Python-side ``@tessera.jit`` failures
    (shape inference, target unavailability, fallback reasons).
  * ``DiagnosticCode`` here covers MLIR-level ``emitOpError`` codes
    produced by C++ verifiers and passes.

This file pins:

  * Every code that appears in ``src/`` C++ code (in the canonical
    ``"CODE_NAME: detail..."`` form) is in the MLIR-verifier registry.
  * Every registered code appears in at least one C++ file.
  * Registry fields are populated correctly (non-empty pass_origin,
    summary, fix_hint; valid severity).

When a future sprint adds a new ``op->emitOpError("FOO_BAR: ...")``
site without registering ``FOO_BAR``, the first test fails with a
clear "unregistered diagnostic code" message — the registry never
silently drifts behind the code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.diagnostic_codes import (
    REGISTERED_CODES,
    all_codes,
    code_lookup,
    codes_by_pass,
    codes_by_sprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


# ─────────────────────────────────────────────────────────────────────────
# C++ scan helpers
# ─────────────────────────────────────────────────────────────────────────


_CODE_PATTERN = re.compile(
    # Match `"NAME_IN_ALL_CAPS:` where NAME is at least 5 chars to avoid
    # false positives on, e.g., MLIR-internal short tokens.
    r'"([A-Z][A-Z0-9_]{4,}):'
)


# Pre-existing C++ tokens that look like our format but aren't MLIR
# verifier codes (e.g., logger tags, debug prefixes, build-config
# tokens that ended up in string literals).  Add specific entries here
# only after manual review confirms they're false positives.
_KNOWN_FALSE_POSITIVES: frozenset[str] = frozenset({
    # Empty for now — the regex (5+ char ALL_CAPS_UNDERSCORE prefix
    # before a colon inside a string literal) is precise enough that no
    # false positives surface today.  When the first one appears,
    # document it here with a comment explaining what it actually is.
})


def _scan_codes_in_cpp() -> dict[str, set[Path]]:
    """Return ``code -> {paths that emit it}`` by scanning every .cpp /
    .h / .mm / .inc under src/."""
    codes: dict[str, set[Path]] = {}
    for ext in ("*.cpp", "*.h", "*.mm", "*.inc"):
        for path in SRC_ROOT.rglob(ext):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for match in _CODE_PATTERN.finditer(text):
                codes.setdefault(match.group(1), set()).add(path)
    return codes


# ─────────────────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────────────────


def test_registered_codes_have_required_fields() -> None:
    """Every entry's required string fields must be non-empty + severity
    is one of the accepted values."""
    for entry in REGISTERED_CODES:
        assert entry.code, f"empty code: {entry}"
        assert entry.pass_origin, f"empty pass_origin for {entry.code}"
        assert entry.summary, f"empty summary for {entry.code}"
        assert entry.fix_hint, f"empty fix_hint for {entry.code}"
        assert entry.sprint, f"empty sprint for {entry.code}"
        assert entry.severity in ("error", "warning"), (
            f"invalid severity {entry.severity!r} for {entry.code}"
        )


def test_registered_codes_have_no_duplicates() -> None:
    """A code must appear at most once in the registry."""
    seen: set[str] = set()
    for entry in REGISTERED_CODES:
        assert entry.code not in seen, (
            f"duplicate code in registry: {entry.code}"
        )
        seen.add(entry.code)


def test_registered_codes_are_alphabetized() -> None:
    """Registry order is alphabetical for easy review.  When adding
    new codes, slot them in sorted position rather than appending."""
    codes = [c.code for c in REGISTERED_CODES]
    assert codes == sorted(codes), (
        "REGISTERED_CODES must be alphabetised; out-of-order codes: "
        f"{[c for c, s in zip(codes, sorted(codes)) if c != s]}"
    )


def test_lookup_helpers_work() -> None:
    """Sanity-check the helper accessors."""
    names = all_codes()
    assert len(names) == len(REGISTERED_CODES)
    assert names == tuple(sorted(names))
    for entry in REGISTERED_CODES:
        looked_up = code_lookup(entry.code)
        assert looked_up is entry
    assert code_lookup("NEVER_REGISTERED_CODE") is None
    sym = codes_by_pass("SymbolicDimEqualityPass")
    assert len(sym) >= 5
    assert all(c.pass_origin == "SymbolicDimEqualityPass" for c in sym)
    v8 = codes_by_sprint("V8")
    assert len(v8) == 6  # 6 Queue codes from V8


# ─────────────────────────────────────────────────────────────────────────
# Drift gates: registry ↔ C++ consistency
# ─────────────────────────────────────────────────────────────────────────


def test_every_cpp_code_is_registered() -> None:
    """Every diagnostic code that appears in C++ source must be in the
    registry.  This is the gate against undocumented codes."""
    cpp_codes = _scan_codes_in_cpp()
    registered = set(all_codes())
    unregistered = {
        code: paths for code, paths in cpp_codes.items()
        if code not in registered and code not in _KNOWN_FALSE_POSITIVES
    }
    if unregistered:
        msg_lines = ["Unregistered MLIR diagnostic codes found in C++ source:"]
        for code, paths in sorted(unregistered.items()):
            relpath = sorted(p.relative_to(REPO_ROOT) for p in paths)[0]
            msg_lines.append(f"  - {code!r} in {relpath}")
        msg_lines.append(
            "Add a DiagnosticCode entry in "
            "`python/tessera/compiler/diagnostic_codes.py` "
            "or add to _KNOWN_FALSE_POSITIVES if it's not actually a "
            "MLIR verifier diagnostic code."
        )
        pytest.fail("\n".join(msg_lines))


def test_every_registered_code_appears_in_cpp() -> None:
    """Every registered code must appear in at least one C++ file.
    Catches stale registry entries left behind after a code rename or
    removal."""
    cpp_codes = _scan_codes_in_cpp()
    missing = [c.code for c in REGISTERED_CODES if c.code not in cpp_codes]
    assert not missing, (
        f"Registered diagnostic codes that don't appear in any C++ file: "
        f"{missing}.  Either restore the C++ emission site or remove the "
        f"stale entry from REGISTERED_CODES."
    )


# ─────────────────────────────────────────────────────────────────────────
# Locked sentinels — these codes are landed across V2 / V3a-c / V4a / V5 / V8
# and MUST stay in the registry.  Catches renames that lose lit-fixture
# coverage.
# ─────────────────────────────────────────────────────────────────────────


_LOCKED_SENTINEL_CODES = (
    "LAYOUT_LEGALITY_UNKNOWN_LAYOUT",
    "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH",
    "SYMDIM_BINDING_VIOLATION",
    "SYMDIM_RESHAPE_VIOLATION",
    "SYMDIM_TRANSPOSE_VIOLATION",
    "SYMDIM_MATMUL_CONTRACT_VIOLATION",
    "SYMDIM_FLOW_INCONSISTENCY",
    "SYMDIM_CALL_ARG_MISMATCH",
    "SYMDIM_LOOP_YIELD_MISMATCH",
    "SYMDIM_IF_BRANCH_MISMATCH",
    "QUEUE_CREATE_OPERAND_COUNT",
    "QUEUE_PUSH_QUEUE_PROVENANCE",
    "QUEUE_PUSH_TILE_TYPE",
    "QUEUE_POP_QUEUE_PROVENANCE",
    "QUEUE_POP_TOKEN_PROVENANCE",
    "QUEUE_POP_TILE_TYPE",
)


@pytest.mark.parametrize("code", _LOCKED_SENTINEL_CODES)
def test_locked_sentinel_code_present(code: str) -> None:
    entry = code_lookup(code)
    assert entry is not None, (
        f"locked sentinel diagnostic code missing: {code}"
    )
    assert entry.severity == "error"
    assert entry.summary, f"empty summary for {code}"
    assert entry.fix_hint, f"empty fix_hint for {code}"
