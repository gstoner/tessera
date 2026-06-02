"""TPP solver Python-side wiring tests.

Locks the Python frontmatter for the TPP solver dialect.  See
``python/tessera/solvers/tpp.py`` for the wired surface and
``src/solvers/tpp/`` for the C++ MLIR side.
"""

from __future__ import annotations

from pathlib import Path

from tessera.solvers import tpp


REPO_ROOT = Path(__file__).resolve().parents[2]
TPP_DIR = REPO_ROOT / "src" / "solvers" / "tpp"


def test_pipeline_alias_name_is_stable() -> None:
    """If we rename ``tpp-space-time`` the C++ ODS and lit fixtures
    have to change too — make that a deliberate decision."""
    assert tpp.TPP_PIPELINE_ALIAS == "tpp-space-time"


def test_pass_names_are_closed_set() -> None:
    """Adding/removing a TPP pass is a deliberate decision."""
    assert tpp.TPP_PASS_NAMES == (
        "tpp-legalize-space-time",
        "tpp-halo-infer",
        "tpp-fuse-stencil-time",
        "tpp-async-prefetch",
        "tpp-vectorize",
        "tpp-distribute-halo",
        "lower-tpp-to-target-ir",
    )


def test_cxx_pass_files_match_python_surface() -> None:
    """Every Python-declared pass must have a corresponding C++ file
    under ``src/solvers/tpp/lib/Passes/``.  Catches drift where the
    C++ side gains/loses a pass without the Python surface
    following."""
    passes_dir = TPP_DIR / "lib" / "Passes"
    cpp_files = {p.name for p in passes_dir.glob("*.cpp")}
    # Map Python pass alias → C++ filename (PascalCase + .cpp).
    expected_cpp = {
        "tpp-legalize-space-time": "LegalizeSpaceTime.cpp",
        "tpp-halo-infer":          "HaloInfer.cpp",
        "tpp-fuse-stencil-time":   "FuseStencilTime.cpp",
        "tpp-async-prefetch":      "AsyncPrefetch.cpp",
        "tpp-vectorize":           "VectorizeTPP.cpp",
        "tpp-distribute-halo":     "DistributeHalo.cpp",
        "lower-tpp-to-target-ir":  "LowerTPPToTargetIR.cpp",
    }
    for pass_alias in tpp.TPP_PASS_NAMES:
        cpp_name = expected_cpp[pass_alias]
        assert cpp_name in cpp_files, (
            f"Python declares pass {pass_alias!r} but "
            f"{passes_dir}/{cpp_name} is missing"
        )


def test_type_and_attr_names() -> None:
    assert tpp.TPP_TYPE_NAMES == ("tpp.field", "tpp.mesh")
    assert tpp.TPP_ATTR_NAMES == ("tpp.units", "tpp.bc")


def test_status_reports_cxx_present_and_embedded_driver_state() -> None:
    """Dialect + passes + alias + lit fixtures always wired. Glass-jaw
    #2 (2026-06-01): ``python_driver_wired`` now tracks whether the
    embedded ``tessera_tpp_capi`` ctypes lib is loadable — True in a
    full build, False in a Python-only checkout. Either way it must
    agree with ``embedded_driver_available()``."""
    s = tpp.status()
    assert s.dialect_present is True
    assert s.passes_present is True
    assert s.pipeline_alias_present is True
    assert s.lit_fixtures_runnable is True
    assert s.python_driver_wired == tpp.embedded_driver_available()
    if s.python_driver_wired:
        assert "in-process" in s.notes.lower()
    else:
        assert "not built" in s.notes.lower() or "ninja" in s.notes


# ── Embedded-MLIR driver (Glass-jaw #2) ───────────────────────────────

def test_solve_raises_without_embedded_lib_or_runs_in_process() -> None:
    """``solve()`` either runs in-process (lib built) or raises a clear
    build hint (lib absent) — never silently subprocesses."""
    if not tpp.embedded_driver_available():
        import pytest
        with pytest.raises(RuntimeError, match="embedded driver unavailable"):
            tpp.solve("module {}\n")
        pytest.skip("tessera_tpp_capi not built in this checkout")
    # Lib present — the canonical pipeline over an empty module returns
    # a module, in-process (no subprocess).
    out = tpp.solve("module {}\n")
    assert "module" in out


def test_solve_runs_tpp_space_time_pipeline_in_process() -> None:
    import pytest
    if not tpp.embedded_driver_available():
        pytest.skip("tessera_tpp_capi not built in this checkout")
    # Default pipeline == tpp-space-time alias.
    out_default = tpp.solve("module {}\n")
    out_explicit = tpp.solve(
        "module {}\n",
        pipeline=f"builtin.module({tpp.TPP_PIPELINE_ALIAS})")
    assert "module" in out_default and "module" in out_explicit


def test_solve_surfaces_pipeline_errors() -> None:
    import pytest
    if not tpp.embedded_driver_available():
        pytest.skip("tessera_tpp_capi not built in this checkout")
    # A bogus pipeline name must surface a RuntimeError, not crash.
    with pytest.raises(RuntimeError):
        tpp.solve("module {}\n",
                  pipeline="builtin.module(this-pass-does-not-exist)")


def test_pipeline_command_shape() -> None:
    cmd = tpp.pipeline_command("foo.mlir")
    assert cmd[0] == "tessera-opt"
    assert any("tpp-space-time" in part for part in cmd)
    assert cmd[-1] == "foo.mlir"


def test_lit_fixtures_present_on_disk() -> None:
    """The four lit fixtures referenced in the README must exist."""
    fixtures = TPP_DIR / "test" / "TPP"
    expected = {
        "halo_infer.mlir",
        "shallow_water_smoke.mlir",
        "bc_lowering.mlir",
        "pipeline_alias.mlir",
    }
    actual = {p.name for p in fixtures.glob("*.mlir")}
    assert expected <= actual, (
        f"missing TPP lit fixtures: {sorted(expected - actual)}"
    )
