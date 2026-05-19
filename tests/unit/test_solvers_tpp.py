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


def test_status_reports_cxx_present_python_driver_not_wired() -> None:
    """Honest reporting: dialect + passes + alias + lit fixtures wired,
    Python embedded-dispatch driver still pending."""
    s = tpp.status()
    assert s.dialect_present is True
    assert s.passes_present is True
    assert s.pipeline_alias_present is True
    assert s.lit_fixtures_runnable is True
    # Honest: ``solve()`` still subprocess-calls ``tessera-opt``
    # instead of dispatching via embedded MLIR Python bindings.
    assert s.python_driver_wired is False
    assert "tessera-opt" in s.notes


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
