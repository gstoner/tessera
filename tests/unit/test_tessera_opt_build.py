"""Smoke tests for the built ``tessera-opt`` binary.

These tests **skip** when ``tessera-opt`` isn't on the PATH or in
the standard build directory.  When the binary is present (i.e.,
after a successful C++ build against MLIR 21), the tests assert
that the headline pipeline aliases and dialects register as
expected.

Closes the M5 follow-up — running these gives the Python-side
visibility into the build state that previously required hand
inspection of ``--help``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _find_tessera_opt() -> str | None:
    """Locate the ``tessera-opt`` binary; ``None`` when unavailable."""
    if DEFAULT_BUILD.is_file() and os.access(DEFAULT_BUILD, os.X_OK):
        return str(DEFAULT_BUILD)
    return shutil.which("tessera-opt")


_TESSERA_OPT = _find_tessera_opt()
_REQUIRES_OPT = pytest.mark.skipif(
    _TESSERA_OPT is None,
    reason="tessera-opt not built; run `cmake --build build --target tessera-opt`",
)


def _run_help() -> str:
    assert _TESSERA_OPT is not None
    out = subprocess.run(
        [_TESSERA_OPT, "--help"], capture_output=True, text=True, timeout=30,
    ).stdout
    return out


@_REQUIRES_OPT
def test_tessera_opt_runs_and_reports_a_version() -> None:
    """Smoke: the binary executes and answers ``--version``."""
    out = subprocess.run(
        [_TESSERA_OPT, "--version"], capture_output=True, text=True, timeout=10,
    ).stdout
    # Homebrew packaging emits "Homebrew LLVM version 21.x.y"; LLVM
    # vanilla emits "LLVM version 21.x.y".  Both contain "21." which is
    # the version pin we care about.
    assert "21." in out, f"unexpected --version output: {out!r}"


@_REQUIRES_OPT
def test_core_tessera_dialects_are_registered() -> None:
    """The Tessera, Neighbors, Solver, and Apple dialects must all
    be registered when ``tessera-opt`` builds with the full feature
    set we ship out of the Apple host."""
    out = _run_help()
    head_line = next((ln for ln in out.splitlines() if "Available Dialects" in ln), "")
    for dialect in ("tessera", "tessera.neighbors", "tessera.solver", "tessera_apple"):
        assert dialect in head_line, (
            f"dialect {dialect!r} missing from tessera-opt --help; "
            f"got: {head_line!r}"
        )


@_REQUIRES_OPT
def test_canonical_pipeline_aliases_are_registered() -> None:
    """Every headline pipeline alias must show up in ``--help``.

    These are the named pass pipelines callers depend on; losing one
    is a contract break.  See ``docs/CANONICAL_API.md``."""
    out = _run_help()
    for alias in (
        "tessera-lower-to-x86",
        "tessera-lower-to-gpu",
        "tessera-lower-to-apple_cpu",
        "tessera-lower-to-apple_cpu-runtime",
        "tessera-lower-to-apple_gpu",
        "tessera-lower-to-apple_gpu-runtime",
    ):
        assert alias in out, f"pipeline alias {alias!r} not registered"


@_REQUIRES_OPT
def test_neighbors_passes_are_registered() -> None:
    """Phase 7 neighbors passes must be on the command line."""
    out = _run_help()
    for pass_name in (
        "tessera-halo-infer",
        "tessera-stencil-lower",
        "tessera-pipeline-overlap",
        "tessera-topology-dynamic",
    ):
        assert pass_name in out, f"neighbors pass {pass_name!r} not registered"


@_REQUIRES_OPT
def test_halo_infer_pass_annotates_stencil_apply(tmp_path: Path) -> None:
    """End-to-end smoke: feed a tiny stencil program through
    ``tessera-halo-infer`` and assert it emits the ``halo.width``
    annotation."""
    src = tmp_path / "halo.mlir"
    src.write_text(
        '''
        func.func @t(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %topo = "tessera.neighbors.topology.create"() {kind = "2d_mesh"}
              : () -> !tessera.neighbors.topology
          %st = "tessera.neighbors.stencil.define"() {
              taps = [dense<[0, 0]> : tensor<2xi64>,
                      dense<[1, 0]> : tensor<2xi64>,
                      dense<[0, 1]> : tensor<2xi64>]
          } : () -> index
          %o = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo)
              : (index, tensor<?x?xf32>, !tessera.neighbors.topology)
                -> tensor<?x?xf32>
          return %o : tensor<?x?xf32>
        }
        ''',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [_TESSERA_OPT, "-allow-unregistered-dialect",
         "-tessera-halo-infer", str(src)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"tessera-opt failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "halo.width = [1, 1]" in proc.stdout, (
        f"halo.width not emitted; stdout was:\n{proc.stdout}"
    )
