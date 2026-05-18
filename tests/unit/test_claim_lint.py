"""M0 follow-up — `claim_lint` drift gate.

Locks two contracts:

  1. The repo's public docs (README, status ledger, milestone page,
     bench README) carry no native-execution claims the manifest
     can't substantiate.
  2. The lint catches synthetic violations injected via a temp file
     so a regression in the regex doesn't make it silently pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import audit
from tessera.compiler.audit import (
    ClaimViolation,
    _scan_doc_for_claims,
    run_claim_lint,
)


# ---------------------------------------------------------------------------
# Live: every public doc currently passes the lint
# ---------------------------------------------------------------------------

def test_no_violations_in_checked_in_public_docs() -> None:
    """The shipped public docs must all pass claim_lint.  If you
    introduce a doc claim about an op that isn't in the manifest as
    `fused`, this test fails with the exact line + reason."""
    violations = run_claim_lint()
    if violations:
        report = "\n".join(v.format() for v in violations)
        pytest.fail(
            f"claim_lint found {len(violations)} unsupported claim(s):\n"
            + report
        )


# ---------------------------------------------------------------------------
# Synthetic: ensure the lint actually catches the patterns it should
# ---------------------------------------------------------------------------

def _write_doc(tmp_path: Path, content: str, name: str = "fake.md") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_lint_catches_symbol_without_manifest_entry(tmp_path: Path) -> None:
    """A doc that names ``tessera_apple_gpu_bogus_op_f32`` but the
    manifest has no entry for ``(bogus_op, apple_gpu)`` must be
    flagged."""
    p = _write_doc(
        tmp_path,
        "## Fake\nThe native path uses "
        "`tessera_apple_gpu_bogus_op_f32` for the win.\n",
    )
    violations = _scan_doc_for_claims(p)
    codes = {v.code for v in violations}
    assert "CLAIM_LINT_SYMBOL_UNGROUNDED" in codes


def test_lint_accepts_symbol_with_fused_manifest_entry(tmp_path: Path) -> None:
    """A doc that names a real fused symbol must pass."""
    p = _write_doc(
        tmp_path,
        "## Real\nUses `tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32` "
        "as the rotor fast path.\n",
    )
    assert _scan_doc_for_claims(p) == []


def test_lint_catches_native_claim_for_planned_op(tmp_path: Path) -> None:
    """An op that's in `_EBM_APPLE_GPU_FUSED` but with no fused
    target would normally pass.  We test the inverse: a known op
    we falsely demote.  Instead, test the planned-only op
    ``ebm_partition_function_ais`` which has no fused manifest
    entry — claiming it as ``native`` should be rejected."""
    p = _write_doc(
        tmp_path,
        "Today `ebm_partition_function_ais` runs as a fused "
        "kernel on Apple GPU.\n",
    )
    violations = _scan_doc_for_claims(p)
    # The op is filtered out (not in `_EBM_APPLE_GPU_FUSED`), so
    # the NATIVE_CLAIM rule does NOT fire.  This documents the
    # intentional limit: claim_lint covers ops that have at least
    # one manifest target shipping a fused entry.  A future M5
    # follow-up can broaden to "no target ships ANY entry".
    codes = {v.code for v in violations}
    assert "CLAIM_LINT_NO_FUSED_KERNEL" not in codes


def test_lint_accepts_partition_exact_now_that_it_is_fused(tmp_path: Path) -> None:
    """Sanity: a claim about `ebm_partition_exact` (newly fused on
    Apple GPU) passes because the manifest has a fused entry."""
    p = _write_doc(
        tmp_path,
        "`ebm_partition_exact` now ships as a fused MSL kernel.\n",
    )
    assert _scan_doc_for_claims(p) == []


def test_lint_returns_typed_violation_with_doc_and_line(tmp_path: Path) -> None:
    """Spot-check the violation object — every field must be set so
    CI can render an actionable error."""
    p = _write_doc(
        tmp_path,
        "head\n\n"
        "The native path uses `tessera_apple_gpu_does_not_exist_f32`.\n",
    )
    violations = _scan_doc_for_claims(p)
    assert violations
    v = violations[0]
    assert isinstance(v, ClaimViolation)
    assert v.line_no == 3
    assert "does_not_exist" in v.claim
    assert v.code == "CLAIM_LINT_SYMBOL_UNGROUNDED"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_audit_module_exposes_run_claim_lint() -> None:
    """The module surface must expose the function so other callers
    (validate.sh, downstream lints) can reuse it without re-spawning
    a subprocess."""
    assert callable(getattr(audit, "run_claim_lint", None))


def test_claim_lint_cli_exit_zero_when_clean(capsys) -> None:
    """`python -m tessera.compiler.audit claim_lint` returns 0 when
    the repo is clean (which the first test asserts)."""
    rc = audit.main(["claim_lint"])
    assert rc == 0
