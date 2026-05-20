"""Python guard tests for the ``tessera-rng-qmcplan`` solver pass
(Phase B2, 2026-05-20).

The pass used to be a 7-line stub that did nothing.  These tests lock
the v1 semantics:

  * ``tessera.solver.method = "quasi_monte_carlo"`` scope ⇒ every
    ``tessera_rng.*`` op gets ``rng.qmc_plan = {seq = "sobol",
    dim_offset = <monotonic>}``.
  * ``tessera.solver.method = "monte_carlo"`` scope (and no scope at
    all) ⇒ ``rng.qmc_plan = {seq = "philox"}`` (no dim_offset, since
    Philox is stateless / counter-based).
  * Module-level ``tessera.rng.qmc_plan_summary`` is stamped with
    ``plans_attached`` + ``sobol_dims`` counters so downstream
    consumers / drift gates can verify the pass actually ran.

The tests run ``tessera-opt --tessera-rng-qmcplan`` as a subprocess.
They skip cleanly when the binary isn't built (the same idiom every
other solver-pass guard test uses — see
``test_spectral_solver_passes.py``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT_CANDIDATES = (
    ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt",
    ROOT / "build-rocm" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_tessera_opt() -> Path | None:
    env_path = os.environ.get("TESSERA_OPT")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for candidate in TESSERA_OPT_CANDIDATES:
        if candidate.exists():
            return candidate
    located = shutil.which("tessera-opt")
    if located:
        return Path(located)
    return None


@pytest.fixture(scope="module")
def tessera_opt() -> Path:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built; skipping RNG QMC plan guard")
    return binary


def _run(binary: Path, mlir: str) -> str:
    proc = subprocess.run(
        [
            str(binary),
            "--tessera-rng-qmcplan",
            "--allow-unregistered-dialect",
            "-",
        ],
        input=mlir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"tessera-opt failed (rc={proc.returncode})\n"
        f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
    return proc.stdout


def test_qmc_scope_attaches_sobol_with_sequential_dim_offset(tessera_opt: Path) -> None:
    """Two RNG ops inside a ``quasi_monte_carlo`` module each receive a
    Sobol sequence assignment with a sequential ``dim_offset`` and the
    module gets a ``plans_attached = 2 / sobol_dims = 2`` summary."""

    mlir = (
        'module attributes {tessera.solver.method = "quasi_monte_carlo"} {\n'
        '  func.func @sobol_kernel() {\n'
        '    "tessera_rng.uniform"() {seed = 42 : i64} : () -> ()\n'
        '    "tessera_rng.normal"() {seed = 42 : i64} : () -> ()\n'
        '    return\n'
        '  }\n'
        '}\n'
    )
    out = _run(tessera_opt, mlir)
    # Two ops, both Sobol, dim_offset = 0 and 1 respectively.
    assert out.count('seq = "sobol"') == 2, out
    assert 'dim_offset = 0 : i64' in out, out
    assert 'dim_offset = 1 : i64' in out, out
    # Module-level summary records the counters.
    assert "plans_attached = 2 : i64" in out, out
    assert "sobol_dims = 2 : i64" in out, out


def test_monte_carlo_scope_attaches_philox(tessera_opt: Path) -> None:
    """``monte_carlo`` scope keeps the legacy Philox path; no
    ``dim_offset`` attribute since Philox is stateless."""

    mlir = (
        'module attributes {tessera.solver.method = "monte_carlo"} {\n'
        '  func.func @mc_kernel() {\n'
        '    "tessera_rng.uniform"() : () -> ()\n'
        '    return\n'
        '  }\n'
        '}\n'
    )
    out = _run(tessera_opt, mlir)
    assert 'seq = "philox"' in out, out
    assert "dim_offset" not in out, (
        "Philox plans should not carry dim_offset (Philox is stateless)"
    )
    assert "plans_attached = 1 : i64" in out, out
    assert "sobol_dims = 0 : i64" in out, out


def test_no_scope_defaults_to_philox(tessera_opt: Path) -> None:
    """A module with no ``tessera.solver.method`` attribute falls back
    to the default Philox sequence rather than guessing at QMC."""

    mlir = (
        "module {\n"
        '  func.func @plain_kernel() {\n'
        '    "tessera_rng.uniform"() : () -> ()\n'
        '    return\n'
        '  }\n'
        '}\n'
    )
    out = _run(tessera_opt, mlir)
    assert 'seq = "philox"' in out, out
    assert "sobol_dims = 0 : i64" in out, out


def test_pass_is_idempotent(tessera_opt: Path) -> None:
    """Running the pass twice produces the same module — an existing
    ``rng.qmc_plan`` attribute is preserved, not overwritten or
    duplicated."""

    mlir = (
        'module attributes {tessera.solver.method = "quasi_monte_carlo"} {\n'
        '  func.func @idem_kernel() {\n'
        '    "tessera_rng.uniform"() : () -> ()\n'
        '  return\n'
        '  }\n'
        '}\n'
    )
    once = _run(tessera_opt, mlir)
    twice = _run(tessera_opt, once)
    # The summary counter goes to 0 on the second pass (already-tagged
    # ops are skipped), but the existing rng.qmc_plan stays put.
    assert 'seq = "sobol"' in twice, twice
    assert "dim_offset = 0 : i64" in twice, twice
    # Second-run summary records zero new attachments.
    assert "plans_attached = 0 : i64" in twice, twice


def test_non_rng_ops_are_untouched(tessera_opt: Path) -> None:
    """The pass only walks ``tessera_rng.*`` ops; other ops (and their
    attributes) must come back identical."""

    # Use ``some_other_dialect.compute`` — fully unregistered so its
    # verifier doesn't reject the empty operand/result list.  The RNG
    # walker keys off the op-name prefix ``tessera_rng.`` and must
    # leave every other op alone, including arbitrary unregistered
    # placeholders that appear in real-world pre-lowering modules.
    mlir = (
        'module attributes {tessera.solver.method = "quasi_monte_carlo"} {\n'
        '  func.func @mixed_kernel() {\n'
        '    "tessera_rng.uniform"() : () -> ()\n'
        '    "some_other_dialect.compute"() : () -> ()\n'
        "    return\n"
        "  }\n"
        "}\n"
    )
    out = _run(tessera_opt, mlir)
    # The unrelated op didn't grow a qmc_plan attr.
    other_line = next(
        line for line in out.splitlines() if "some_other_dialect.compute" in line
    )
    assert "qmc_plan" not in other_line, other_line
    # The RNG op did.
    rng_line = next(
        line for line in out.splitlines() if "tessera_rng.uniform" in line
    )
    assert "qmc_plan" in rng_line, rng_line
