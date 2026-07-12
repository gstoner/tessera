"""Followup 2 — ``--verify-fixtures`` CLI mode for the op×target
conformance dashboard.

A.3 wired ``execute_compare_fixture`` paths into ``manifest_for(op)``
so the dashboard's ``numerical_check`` column reflects manifest-
declared truth. This CLI mode turns that declaration into actual
proof: invoke pytest on every declared fixture and report per-cell
pass/fail. Replaces "the dashboard says this is verified" with "we
just verified it."

These tests pin the CLI's structural contract — they DO NOT re-run
all 11 declared fixtures (that's the job of the CLI itself in CI). We
exercise:

1. The argparse surface (``--verify-fixtures`` is mutually exclusive
   with ``--render`` / ``--check``).
2. The de-duplication shape — a single test file may cover multiple
   (op, target) pairs; we should invoke it once and report against
   every pair.
3. The exit-code contract — non-zero when any fixture fails; zero
   when all pass.
4. A live smoke that runs the CLI on a single fixture and confirms
   it reports a result for every (op, target) pair the fixture
   covers.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from tessera.cli import conformance_matrix as cli_cm
from tessera.compiler import backend_manifest as bm


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- argparse surface ---------------------------------------------------

def test_verify_fixtures_flag_is_mutually_exclusive_with_render():
    """``--verify-fixtures --render`` should fail argparse — the three
    modes are exclusive."""
    err = io.StringIO()
    with redirect_stderr(err):
        with pytest.raises(SystemExit):
            cli_cm.main(["--verify-fixtures", "--render"])


def test_no_mode_flag_is_required():
    """At least one of the three mode flags must be present."""
    err = io.StringIO()
    with redirect_stderr(err):
        with pytest.raises(SystemExit):
            cli_cm.main([])


# ---- exit-code contract -------------------------------------------------

@pytest.fixture(scope="module")
def real_verify_result():
    """Run the real fixture verifier ONCE and share its (rc, output).

    ``_verify_fixtures()`` invokes pytest on every declared fixture (~a minute),
    so the three contract checks that exercise a real run must not each pay that
    cost — doing so was ~3×66s ≈ 200s, ~20% of the whole unit suite. The two
    *patched* tests below (bogus map / mocked subprocess) deliberately do NOT use
    this fixture: they need their own state and are already fast."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_cm._verify_fixtures()
    return rc, buf.getvalue()


@pytest.mark.slow
def test_verify_fixtures_returns_zero_when_all_pass(real_verify_result):
    """The dashboard ships 11 declared fixtures, all numerically
    correct on this Mac as of audit-followup ship time. Run the
    real verifier and assert exit code 0.

    Marked ``slow``: the ``real_verify_result`` fixture invokes a NESTED pytest
    over every declared execute-compare fixture (~300s in this suite) — but each
    of those fixtures already runs as a first-class test in the same sweep, so
    this only re-verifies the CLI *plumbing* returns 0. The failure path (CLI
    returns non-zero) is covered fast by
    ``test_verify_fixtures_returns_nonzero_when_any_fixture_fails`` (mocked), and
    the fixture-proof discipline itself by ``test_conformance_complete_cells_
    proven.py``. Keeping it out of the parallel unit lane removes the single
    biggest straggler (it capped even a 32-core `-n auto` run)."""
    rc, output = real_verify_result
    assert rc == 0, f"verify-fixtures returned {rc}; output was:\n{output}"
    assert "FAIL" not in output.split("summary:")[0], (
        "summary reported 0 failures but per-fixture log shows FAIL")


def test_verify_fixtures_returns_nonzero_when_any_fixture_fails():
    """Patch the fixture map to point at a non-existent file and
    confirm the verifier exits non-zero with a clean diagnostic."""
    bogus_map = {("matmul", "cpu"): "tests/unit/__does_not_exist__.py"}
    with patch.object(bm, "_NUMERICAL_FIXTURES", bogus_map):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_cm._verify_fixtures()
        assert rc == 1, f"expected non-zero exit; got {rc}"
        assert "fixture file missing" in buf.getvalue()


# ---- de-duplication / coverage shape ------------------------------------

def test_fixtures_are_run_once_per_file_not_per_pair():
    """``test_apple_gpu_mpsgraph_lane.py`` covers 5 (op, target) pairs.
    The CLI must invoke pytest ONCE on that file and report the same
    result against all 5 pairs — not invoke 5 separate pytest runs."""
    invocations: list[list[str]] = []

    real_run = subprocess.run

    def _capture(cmd, *args, **kwargs):
        invocations.append(list(cmd))
        # Force-pass to keep the test self-contained — we only care
        # about how MANY pytest invocations happen.
        return real_run([sys.executable, "-c", "import sys; sys.exit(0)"],
                        capture_output=True, text=True)

    with patch("subprocess.run", side_effect=_capture):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_cm._verify_fixtures()

    # Number of subprocess.run calls = number of unique fixture files.
    unique_files = {rel for rel in bm._NUMERICAL_FIXTURES.values()}
    pytest_calls = [c for c in invocations if "pytest" in " ".join(c)]
    assert len(pytest_calls) == len(unique_files), (
        f"expected {len(unique_files)} pytest invocations (one per file); "
        f"got {len(pytest_calls)}")


# ---- summary reporting --------------------------------------------------

@pytest.mark.slow
def test_summary_line_reports_total_pair_count(real_verify_result):
    """The summary line must enumerate every declared (op, target)
    pair — not just unique files."""
    _rc, output = real_verify_result
    n_pairs = len(bm._NUMERICAL_FIXTURES)
    # The header line names the pair count.
    assert f"{n_pairs} declared (op, target) pair" in output, output


@pytest.mark.slow
def test_per_fixture_line_names_covered_pairs(real_verify_result):
    """The per-file line must list which (op, target) pairs each
    invocation covers. Catches a regression where the CLI runs pytest
    once but loses the pair-attribution mapping."""
    _rc, output = real_verify_result
    # Every (op, target) pair must appear in some per-file line.
    for (op, target) in bm._NUMERICAL_FIXTURES:
        assert f"{op}/{target}" in output, (
            f"pair {op}/{target} missing from per-fixture report:\n{output}")
