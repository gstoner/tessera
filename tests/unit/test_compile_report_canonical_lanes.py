"""Slice 4 (2026-05-22) — CompileReport canonical schema meta-test.

The "everyday truth source" gate: every shipped JIT lane + every
canonical program in ``tessera.compiler.canonical`` must emit a
:class:`CompileReport` with a stable schema, so frontend/JIT/textual/
GA/EBM status can't drift apart again.

What this test locks:

  1. Every canonical program in ``python/tessera/compiler/canonical/``
     either exposes a ``run()`` (or equivalent entry point) that
     returns a CompileReport, OR is justified-and-documented as not
     a report-emitting program.
  2. Every shipped JIT callable wrapper (``CliffordCompiledCallable``,
     ``EnergyCompiledCallable``) exposes ``.compile_report()``
     returning a valid CompileReport.
  3. All reports validate against the dataclass invariants (frontend
     ∈ whitelist, value_kind ∈ whitelist, ir_layers ∈ whitelist).
  4. The frontend label whitelist is exactly three — adding a fourth
     requires an explicit update here (preventing accidental
     "frontend bloat" that would split status apart).
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import compile_report as _cr
from tessera.compiler.compile_report import (
    CompileReport,
    FRONTEND_CLIFFORD_JIT,
    FRONTEND_TESSERA_JIT,
    FRONTEND_TEXTUAL,
    IR_LAYERS,
    VALID_FRONTENDS,
    VALID_VALUE_KINDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = (
    REPO_ROOT / "python" / "tessera" / "compiler" / "canonical"
)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical schema invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_frontend_whitelist_is_exactly_three() -> None:
    """The three blessed lanes: @tessera.jit, textual, @clifford_jit.

    Adding a 4th frontend means the lane is no longer constrained to
    the AST-template umbrella — that's a deliberate architectural
    change that must update this test in lockstep.

    energy_jit and complex_jit funnel through the clifford_jit AST
    template and report with ``frontend=clifford_jit`` + a
    disambiguating ``source`` field.  This keeps the lane count low
    without sacrificing per-lane provenance.
    """
    assert VALID_FRONTENDS == {
        FRONTEND_TESSERA_JIT, FRONTEND_TEXTUAL, FRONTEND_CLIFFORD_JIT,
    }


def test_ir_layers_is_canonical_four() -> None:
    assert IR_LAYERS == ("graph_ir", "schedule_ir", "tile_ir", "target_ir")


# ─────────────────────────────────────────────────────────────────────────────
# Every canonical program emits a valid CompileReport
# ─────────────────────────────────────────────────────────────────────────────


CANONICAL_PROGRAMS = (
    # name, module path, run-fn name, run-kwargs
    ("conv2d_norm_activation",
     "tessera.compiler.canonical.conv2d_norm_activation",
     "run", {}),
    ("decode_init_inner_loop_self_verify",
     "tessera.compiler.canonical.decode_init_inner_loop_self_verify",
     "run", {}),
    ("kv_cache_append_prune_read",
     "tessera.compiler.canonical.kv_cache_append_prune_read",
     "run", {}),
    ("matmul_gelu",
     "tessera.compiler.canonical.matmul_gelu",
     "run", {}),
    ("matmul_softmax_matmul",
     "tessera.compiler.canonical.matmul_softmax_matmul",
     "run", {}),
    ("rotor_sandwich_ebt_tiny",
     "tessera.compiler.canonical.rotor_sandwich_ebt_tiny",
     "run", {}),
    ("rotor_sandwich_norm",
     "tessera.compiler.canonical.rotor_sandwich_norm",
     "run", {}),
    ("visual_complex_fused",
     "tessera.compiler.canonical.visual_complex_fused",
     "run", {}),
)


@pytest.mark.parametrize(
    "name,module_path,run_fn_name,run_kwargs",
    CANONICAL_PROGRAMS,
    ids=[p[0] for p in CANONICAL_PROGRAMS],
)
def test_canonical_program_emits_valid_compile_report(
    name: str, module_path: str, run_fn_name: str, run_kwargs: dict,
) -> None:
    """For each canonical program, ``run()`` returns a CompileReport
    with all required schema fields populated."""
    mod = importlib.import_module(module_path)
    run = getattr(mod, run_fn_name, None)
    assert run is not None, f"{module_path}.{run_fn_name} missing"
    report = run(**run_kwargs)
    assert isinstance(report, CompileReport), (
        f"{name}: run() returned {type(report).__name__}, expected CompileReport"
    )
    # Schema invariants — the dataclass enforces these in __post_init__,
    # but be explicit so a regression to non-canonical strings surfaces
    # here, not from a downstream cryptic dataclass error.
    assert report.frontend in VALID_FRONTENDS
    assert report.value_kind in VALID_VALUE_KINDS
    assert report.target  # non-empty target string
    assert report.program_id  # non-empty
    # Every canonical program records a graph_ir hash.
    assert "graph_ir" in report.ir_hashes, (
        f"{name}: missing graph_ir digest"
    )
    # The hash is stable + deterministic (same input → same hash).
    h1 = report.report_hash()
    h2 = report.report_hash()
    assert h1 == h2, f"{name}: report_hash non-deterministic"


def test_all_eight_canonical_programs_covered() -> None:
    """If a 9th canonical program lands, this test fails until the
    new program is added to CANONICAL_PROGRAMS.  Catches the bug
    class 'new program added, meta-test forgot to include it'."""
    py_files = sorted(
        f.name for f in CANONICAL_DIR.glob("*.py")
        if f.name != "__init__.py"
    )
    covered = sorted(f"{p[0]}.py" for p in CANONICAL_PROGRAMS)
    assert covered == py_files, (
        f"canonical/ has {py_files} but CANONICAL_PROGRAMS covers {covered}; "
        "if a new program landed, add it to CANONICAL_PROGRAMS"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Every shipped JIT callable wrapper exposes .compile_report()
# ─────────────────────────────────────────────────────────────────────────────


def test_clifford_compiled_callable_emits_report() -> None:
    """``CliffordCompiledCallable.compile_report()`` returns a valid
    CompileReport (already shipped — this test pins the contract)."""
    from tessera.compiler.clifford_jit import (
        clifford_jit, CliffordCompiledCallable,
    )
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f(a, b):
        return ga.inner(a, b)

    assert isinstance(f, CliffordCompiledCallable)
    report = f.compile_report()
    assert isinstance(report, CompileReport)
    assert report.frontend == FRONTEND_CLIFFORD_JIT
    assert report.value_kind in VALID_VALUE_KINDS
    assert "@clifford_jit" in report.source


def test_energy_compiled_callable_emits_report() -> None:
    """``EnergyCompiledCallable.compile_report()`` is the Slice 4 add
    — energy_jit lanes now expose the same accessor as clifford_jit
    so cross-lane consumers can iterate uniformly."""
    from tessera.compiler.energy_jit import (
        energy_jit, EnergyCompiledCallable,
    )
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y):
        q = energy.norm_sq(y)
        r = energy.relu(y)
        out = energy.inner(r, y)
        return energy.reduce_sum(out)

    assert isinstance(E, EnergyCompiledCallable)
    report = E.compile_report()
    assert isinstance(report, CompileReport)
    # Per the AST-constrained-lane convention, the frontend is
    # clifford_jit (the AST-template lane) and the source field
    # disambiguates "energy_jit".
    assert report.frontend == FRONTEND_CLIFFORD_JIT
    assert report.value_kind == _cr.VALUE_KIND_TENSOR
    assert "@energy_jit" in report.source
    # The IR digest is populated.
    assert "graph_ir" in report.ir_hashes
    # The target_decision documents this came from energy_jit.
    assert "apple_gpu" in report.target_decision
    assert "energy_jit" in report.target_decision["apple_gpu"]


def test_energy_compile_report_deterministic_across_runs() -> None:
    """Same energy function decorated twice must produce identical
    report hashes — the stability invariant that makes the report a
    useful drift gate."""
    from tessera.compiler.energy_jit import energy_jit
    from tessera import energy

    def make() -> CompileReport:
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)
        return E.compile_report()

    r1 = make()
    r2 = make()
    assert r1.report_hash() == r2.report_hash(), (
        "energy_jit CompileReport must be stable across decoration runs"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-lane consistency — the schema is the same across lanes
# ─────────────────────────────────────────────────────────────────────────────


def test_clifford_and_energy_reports_share_schema() -> None:
    """Any consumer (status dashboard, audit walker, CI report
    serialiser) must be able to treat both ``CliffordCompiledCallable``
    and ``EnergyCompiledCallable`` reports uniformly."""
    from tessera.compiler.clifford_jit import clifford_jit
    from tessera.compiler.energy_jit import energy_jit
    import tessera.ga as ga
    from tessera import energy

    @clifford_jit(target="apple_gpu")
    def cf(a, b):
        return ga.inner(a, b)

    @energy_jit(target="apple_gpu")
    def ef(y):
        return energy.norm_sq(y)

    cr = cf.compile_report()
    er = ef.compile_report()
    # Both serialise to dict.
    cd = cr.as_dict()
    ed = er.as_dict()
    # Same schema keys (CompileReport invariant).
    assert set(cd.keys()) == set(ed.keys()), (
        f"schema drift: clifford={set(cd.keys()) - set(ed.keys())}, "
        f"energy={set(ed.keys()) - set(cd.keys())}"
    )
    # Both report via the clifford_jit AST-lane convention.
    assert cd["frontend"] == ed["frontend"] == FRONTEND_CLIFFORD_JIT
    # Both round-trip through JSON.
    import json
    assert json.dumps(cd)
    assert json.dumps(ed)
