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
    codes_by_language,
    codes_by_pass,
    codes_by_sprint,
    codes_by_status,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PYTHON_TESSERA_ROOT = REPO_ROOT / "python" / "tessera"


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


# TSOL-2 (2026-05-22): scan Python source for E_* / JIT_* / TS_ERR_*
# emissions.  The pattern matches enum values + string-literal raises;
# we keep the same shape as the C++ scan so the drift gate stays
# symmetric.
_PYTHON_CODE_PATTERNS = (
    re.compile(r'"(E_[A-Z][A-Z0-9_]{1,})"'),
    re.compile(r'"(JIT_[A-Z][A-Z0-9_]{2,})"'),
    re.compile(r'"(TS_ERR_[A-Z][A-Z0-9_]{2,})"'),
)


def _scan_codes_in_python() -> dict[str, set[Path]]:
    """Return ``code -> {paths that emit it}`` by scanning every .py
    file under python/tessera/.  Excludes the diagnostic_codes.py
    registry itself (it lists every code by definition) so the gate
    measures real emission sites, not the manifest."""
    codes: dict[str, set[Path]] = {}
    registry_file = (
        PYTHON_TESSERA_ROOT / "compiler" / "diagnostic_codes.py"
    )
    for path in PYTHON_TESSERA_ROOT.rglob("*.py"):
        if path == registry_file:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for pattern in _PYTHON_CODE_PATTERNS:
            for match in pattern.finditer(text):
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


def test_registered_codes_group_by_language_and_prefix() -> None:
    """The registry organises codes by language (mlir / python) and
    by prefix family (E_*, JIT_*, TS_ERR_*, SYMDIM_*, QUEUE_*,
    LAYOUT_LEGALITY_*).  We don't enforce strict alphabetisation
    across the registry (low-value churn when adding new prefixes);
    we do verify the prefix-to-language mapping stays sensible."""
    # Each prefix maps to exactly one language.
    prefix_to_lang: dict[str, str] = {}
    for c in REGISTERED_CODES:
        prefix = c.code.split("_", 1)[0]
        if prefix in prefix_to_lang:
            assert prefix_to_lang[prefix] == c.language, (
                f"prefix {prefix!r} declared in both "
                f"{prefix_to_lang[prefix]!r} and {c.language!r} languages"
            )
        else:
            prefix_to_lang[prefix] = c.language
    # Canonical mapping (locked sentinels).
    expected = {
        "E": "python",
        "JIT": "python",
        "TS": "python",          # TS_ERR_* codes
        "SYMDIM": "mlir",
        "QUEUE": "mlir",
        "LAYOUT": "mlir",         # LAYOUT_LEGALITY_*
    }
    for prefix, lang in expected.items():
        assert prefix_to_lang.get(prefix) == lang, (
            f"prefix {prefix!r} expected to be {lang!r} language, got "
            f"{prefix_to_lang.get(prefix)!r}"
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
    """Every MLIR diagnostic code that appears in C++ source must be
    in the registry.  This is the gate against undocumented MLIR-side
    codes."""
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


def test_every_mlir_registered_code_appears_in_cpp() -> None:
    """Every MLIR-language registered code must appear in at least one
    C++ file.  Catches stale registry entries left behind after a
    code rename or removal."""
    cpp_codes = _scan_codes_in_cpp()
    mlir_codes = [c for c in REGISTERED_CODES if c.language == "mlir"]
    missing = [c.code for c in mlir_codes if c.code not in cpp_codes]
    assert not missing, (
        f"Registered MLIR diagnostic codes that don't appear in any "
        f"C++ file: {missing}.  Either restore the C++ emission site "
        f"or remove the stale entry from REGISTERED_CODES."
    )


# ─────────────────────────────────────────────────────────────────────────
# TSOL-2 (2026-05-22) — Python-side drift gates
# ─────────────────────────────────────────────────────────────────────────


def test_every_python_code_is_registered() -> None:
    """Every E_* / JIT_* / TS_ERR_* code that appears in Python source
    (outside the registry itself) must be in the registry."""
    py_codes = _scan_codes_in_python()
    registered = set(all_codes())
    unregistered = {
        code: paths for code, paths in py_codes.items()
        if code not in registered
    }
    if unregistered:
        msg_lines = ["Unregistered Python diagnostic codes found:"]
        for code, paths in sorted(unregistered.items()):
            relpath = sorted(p.relative_to(REPO_ROOT) for p in paths)[0]
            msg_lines.append(f"  - {code!r} in {relpath}")
        msg_lines.append(
            "Add a DiagnosticCode entry with language='python' in "
            "`python/tessera/compiler/diagnostic_codes.py`."
        )
        pytest.fail("\n".join(msg_lines))


def test_python_implemented_codes_appear_in_python_source() -> None:
    """Every Python-language registered code with status='implemented'
    must appear in at least one .py file.  Codes with
    status='spec_contract' are exempt — they document a TSOL contract
    that hasn't been wired into Python emission sites yet."""
    py_codes = _scan_codes_in_python()
    implemented_python = [
        c for c in REGISTERED_CODES
        if c.language == "python" and c.status == "implemented"
    ]
    missing = [c.code for c in implemented_python if c.code not in py_codes]
    assert not missing, (
        f"Registered Python codes claimed as 'implemented' that don't "
        f"appear in any .py file: {missing}.  Either flip their status "
        f"to 'spec_contract' or restore the Python emission site."
    )


def test_codes_by_language_helper() -> None:
    mlir = codes_by_language("mlir")
    python = codes_by_language("python")
    # All codes are partitioned cleanly across the two languages.
    assert len(mlir) + len(python) == len(REGISTERED_CODES)
    # Each MLIR code's pass_origin reflects an MLIR pass / verifier;
    # not a Python module.
    for c in mlir:
        assert "tessera.compiler" not in c.pass_origin, (
            f"MLIR code {c.code!r} claims a Python pass_origin"
        )


def test_codes_by_status_helper() -> None:
    impl = codes_by_status("implemented")
    contracts = codes_by_status("spec_contract")
    assert len(impl) + len(contracts) == len(REGISTERED_CODES)
    # Today's TSOL-2 baseline: exactly 6 TS_ERR_* codes are
    # spec_contract.
    ts_err = [c for c in contracts if c.code.startswith("TS_ERR_")]
    assert len(ts_err) == 6, (
        f"Expected 6 TS_ERR_* spec contracts; got {len(ts_err)}"
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
    # TSOL-2 (2026-05-22) — Python-side family sentinels.
    "E_SHAPE_MISMATCH",
    "E_TILE_LOWERING",
    "E_TARGET_CODEGEN",
    "JIT_SOURCE_UNAVAILABLE",
    "JIT_SOURCE_PROVIDED",
    "JIT_EAGER_FALLBACK_EMPTY",
    "JIT_EAGER_FALLBACK_UNSUPPORTED_OP",
    "JIT_EAGER_FALLBACK_ARITY",
    "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY",
    # A.2 (2026-05-31): scf.* structured control flow gets its own
    # info-level code (distinct from the generic unknown-op miss) so
    # dashboards can surface it as an expected eager path.
    "JIT_EAGER_FALLBACK_CONTROL_FLOW",
    "JIT_COMPILED_CPU",
    "JIT_COMPILED_TARGET_RUNTIME",
    "JIT_TARGET_IR_ARTIFACT_ONLY",
    "TS_ERR_INVALID_ARG",
    "TS_ERR_SHAPE_MISMATCH",
    "TS_ERR_UNSUPPORTED_DTYPE",
    "TS_ERR_BACKEND_FAILURE",
    "TS_ERR_OOM",
    "TS_ERR_NONDETERMINISM",
)


@pytest.mark.parametrize("code", _LOCKED_SENTINEL_CODES)
def test_locked_sentinel_code_present(code: str) -> None:
    entry = code_lookup(code)
    assert entry is not None, (
        f"locked sentinel diagnostic code missing: {code}"
    )
    # JIT_* codes are legitimately warnings (informational telemetry);
    # E_* / SYMDIM_* / QUEUE_* / LAYOUT_LEGALITY_* / TS_ERR_* are errors.
    assert entry.severity in ("error", "warning"), (
        f"invalid severity for {code}: {entry.severity!r}"
    )
    assert entry.summary, f"empty summary for {code}"
    assert entry.fix_hint, f"empty fix_hint for {code}"
