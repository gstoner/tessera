"""Smoke tests for the built ``tessera-opt`` binary.

These tests **skip** when ``tessera-opt`` isn't on the PATH or in
the standard build directory.  When the binary is present (i.e.,
after a successful C++ build against MLIR 22), the tests assert
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
DEFAULT_BUILD_CANDIDATES = (
    REPO_ROOT / "build-llvm22" / "tools" / "tessera-opt" / "tessera-opt",
    REPO_ROOT / "build-llvm22-make" / "tools" / "tessera-opt" / "tessera-opt",
    REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_tessera_opt() -> str | None:
    """Locate the ``tessera-opt`` binary; ``None`` when unavailable."""
    if explicit := os.environ.get("TESSERA_OPT_PATH"):
        candidate = Path(explicit)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    for candidate in DEFAULT_BUILD_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("tessera-opt")


_TESSERA_OPT = _find_tessera_opt()
_REQUIRES_OPT = pytest.mark.skipif(
    _TESSERA_OPT is None,
    reason="tessera-opt not built; run `cmake --build build-llvm22 --target tessera-opt`",
)


def _run_help() -> str:
    assert _TESSERA_OPT is not None
    out = subprocess.run(
        [_TESSERA_OPT, "--help"], capture_output=True, text=True, timeout=30,
    ).stdout
    return out


def _available_dialects(help_text: str) -> set[str]:
    head_line = next(
        (ln for ln in help_text.splitlines() if "Available Dialects" in ln),
        "",
    )
    if ":" not in head_line:
        return set()
    return {name.strip() for name in head_line.split(":", 1)[1].split(",")}


@_REQUIRES_OPT
def test_tessera_opt_runs_and_reports_a_version() -> None:
    """Smoke: the binary executes and answers ``--version``."""
    out = subprocess.run(
        [_TESSERA_OPT, "--version"], capture_output=True, text=True, timeout=10,
    ).stdout
    # Homebrew packaging emits "Homebrew LLVM version 22.x.y"; LLVM
    # vanilla emits "LLVM version 22.x.y".  Both contain "22." which is
    # the version pin we care about.
    assert "22." in out, f"unexpected --version output: {out!r}"


@_REQUIRES_OPT
def test_core_tessera_dialects_are_registered() -> None:
    """The Tessera, Neighbors, Solver, Apple, and TPP dialects must
    all be registered when ``tessera-opt`` builds with the full
    feature set we ship out of the Apple host."""
    out = _run_help()
    dialects = _available_dialects(out)
    for dialect in (
        "tessera",
        "tessera.neighbors",
        "tessera.solver",
        "tpp",
    ):
        assert dialect in dialects, (
            f"dialect {dialect!r} missing from tessera-opt --help; "
            f"got: {sorted(dialects)!r}"
        )
    if "tessera_apple" not in dialects:
        return


@_REQUIRES_OPT
def test_tpp_passes_and_pipeline_alias_are_registered() -> None:
    """All 7 TPP individual passes and the ``tpp-space-time``
    pipeline alias must show up in ``--help``."""
    out = _run_help()
    for pass_name in (
        "tpp-legalize-space-time",
        "tpp-halo-infer",
        "tpp-fuse-stencil-time",
        "tpp-async-prefetch",
        "tpp-vectorize",
        "tpp-distribute-halo",
        "lower-tpp-to-target-ir",
        "tpp-space-time",   # pipeline alias
    ):
        assert pass_name in out, (
            f"TPP pass / alias {pass_name!r} not registered"
        )


@_REQUIRES_OPT
def test_tpp_space_time_pipeline_runs(tmp_path: Path) -> None:
    """End-to-end: feed a tiny TPP program through ``tpp-space-time``
    and confirm the binary produces valid MLIR output."""
    src = tmp_path / "tpp.mlir"
    src.write_text(
        '''
        func.func @halo_example(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
          %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
          return %y : tensor<32x32xf32>
        }
        ''',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [_TESSERA_OPT, "-allow-unregistered-dialect",
         "-tpp-space-time", str(src)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"tessera-opt failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "tpp.halo" in proc.stdout, (
        f"tpp-halo-infer step didn't run; stdout was:\n{proc.stdout}"
    )


@_REQUIRES_OPT
def test_canonical_pipeline_aliases_are_registered() -> None:
    """Every headline pipeline alias must show up in ``--help``.

    These are the named pass pipelines callers depend on; losing one
    is a contract break.  See ``docs/CANONICAL_API.md``."""
    out = _run_help()
    dialects = _available_dialects(out)
    for alias in (
        "tessera-lower-to-x86",
        "tessera-lower-to-gpu",
    ):
        assert alias in out, f"pipeline alias {alias!r} not registered"
    if "tessera_apple" not in dialects:
        return
    for alias in (
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


_DEFAULT_TRANSLATE_BUILD = (
    REPO_ROOT / "build-llvm22" / "tools" / "tessera-translate" / "tessera-translate-mlir",
    REPO_ROOT / "build-llvm22-make" / "tools" / "tessera-translate" / "tessera-translate-mlir",
    REPO_ROOT / "build" / "tools" / "tessera-translate" / "tessera-translate-mlir"
)


def _find_tessera_translate_mlir() -> str | None:
    for candidate in _DEFAULT_TRANSLATE_BUILD:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    path_candidate = shutil.which("tessera-translate-mlir")
    if path_candidate is None:
        return None
    out = subprocess.run(
        [path_candidate, "--version"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    if "22." not in out:
        return None
    return path_candidate


_TRANSLATE_MLIR = _find_tessera_translate_mlir()
_REQUIRES_TRANSLATE = pytest.mark.skipif(
    _TRANSLATE_MLIR is None,
    reason=(
        "tessera-translate-mlir not built; run "
        "`cmake --build build --target tessera-translate-mlir`"
    ),
)


@_REQUIRES_TRANSLATE
def test_tessera_translate_mlir_runs_and_reports_version() -> None:
    out = subprocess.run(
        [_TRANSLATE_MLIR, "--version"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    assert "22." in out, f"unexpected --version output: {out!r}"


@_REQUIRES_TRANSLATE
def test_tessera_translate_mlir_translates_llvm_dialect_to_llvm_ir(
    tmp_path: Path,
) -> None:
    """End-to-end: a tiny LLVM-dialect module → LLVM IR text."""
    src = tmp_path / "add.mlir"
    src.write_text(
        '''
        module {
          llvm.func @add(%a: i32, %b: i32) -> i32 {
            %0 = llvm.add %a, %b : i32
            llvm.return %0 : i32
          }
        }
        ''',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [_TRANSLATE_MLIR, "--mlir-to-llvmir", str(src)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"tessera-translate-mlir failed:\nstdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )
    # The translation produces real LLVM IR text.
    assert "define i32 @add" in proc.stdout
    assert "= add i32" in proc.stdout
    assert "ret i32" in proc.stdout


@_REQUIRES_TRANSLATE
def test_tessera_translate_mlir_advertises_documented_flags() -> None:
    """The four translation flags the README/source comment promise
    must all appear in --help."""
    out = subprocess.run(
        [_TRANSLATE_MLIR, "--help"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    for flag in (
        "--mlir-to-llvmir",
        "--import-llvm",
        "--serialize-spirv",
        "--deserialize-spirv",
    ):
        assert flag in out, f"documented flag {flag!r} missing from --help"


@_REQUIRES_TRANSLATE
def test_tessera_translate_mlir_serializes_and_deserializes_spirv(
    tmp_path: Path,
) -> None:
    """SPIR-V round-trip: spirv.module text → .spv binary → spirv.module text."""
    src = tmp_path / "shader.mlir"
    src.write_text(
        '''
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @main() "None" {
    spirv.Return
  }
  spirv.EntryPoint "Vertex" @main
}
''',
        encoding="utf-8",
    )
    spv = tmp_path / "shader.spv"
    # Serialize.  `--no-implicit-module` is required because the input is
    # a top-level `spirv.module`, not nested inside a `builtin.module`.
    ser = subprocess.run(
        [_TRANSLATE_MLIR, "--serialize-spirv", "--no-implicit-module",
         str(src), "-o", str(spv)],
        capture_output=True, text=True, timeout=30,
    )
    assert ser.returncode == 0, (
        f"serialize-spirv failed:\nstdout: {ser.stdout}\nstderr: {ser.stderr}"
    )
    assert spv.is_file() and spv.stat().st_size > 0
    # SPIR-V binary magic number 0x07230203 (little-endian: 03 02 23 07).
    assert spv.read_bytes()[:4] == bytes([0x03, 0x02, 0x23, 0x07]), (
        f"output is not a SPIR-V binary; head: {spv.read_bytes()[:8]!r}"
    )

    # Deserialize back.
    deser = subprocess.run(
        [_TRANSLATE_MLIR, "--deserialize-spirv", str(spv)],
        capture_output=True, text=True, timeout=30,
    )
    assert deser.returncode == 0, (
        f"deserialize-spirv failed:\nstdout: {deser.stdout}\nstderr: {deser.stderr}"
    )
    assert "spirv.module" in deser.stdout
    assert "spirv.func @main" in deser.stdout
    assert "spirv.EntryPoint" in deser.stdout


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
