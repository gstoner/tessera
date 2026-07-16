"""Gap 2 — real schedule.prefetch overlap pass (was a no-op stub).

`tpp-async-prefetch` now software-pipelines `schedule.prefetch` ops: rotating
double-buffer stages + dependency-safe hoist of overlap-policy prefetches above
preceding compute. An ``into="host"`` / ``overlap="none"`` prefetch — the way
LSA cold-pool staging is *recorded* — is annotated but never overlapped/hoisted
(no overlap semantics claimed). See `docs/audit/domain/archive/lsa_scope.md`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "solvers" / "tpp" / "lib" / "Passes" / "AsyncPrefetch.cpp"
_CANDIDATES = (
    REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt",
    REPO_ROOT / "build-llvm23" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_opt():
    if explicit := os.environ.get("TESSERA_OPT_PATH"):
        if Path(explicit).is_file():
            return explicit
    for c in _CANDIDATES:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
_needs_opt = pytest.mark.skipif(_OPT is None, reason="tessera-opt not built")

_FIXTURE = '''
func.func @pipe(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "comp.matmul"(%a) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %p0 = "schedule.prefetch"(%b) {into = "shared", overlap = "compute"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %p1 = "schedule.prefetch"(%b) {into = "shared", overlap = "compute"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %p2 = "schedule.prefetch"(%c) {into = "host", overlap = "none"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %p1 : tensor<4x4xf32>
}
'''


def test_pass_body_is_not_a_stub():
    src = SRC.read_text()
    assert "runOnOperation() final {}" not in src, "AsyncPrefetch is still a no-op"
    assert "(stub)" not in src
    assert "tpp.prefetch.overlapped" in src
    assert "moveBefore" in src  # the hoist that realizes overlap


@_needs_opt
def test_overlap_prefetch_is_staged_and_hoisted(tmp_path):
    f = tmp_path / "pf.mlir"
    f.write_text(_FIXTURE)
    out = subprocess.run(
        [_OPT, str(f), "-tpp-async-prefetch", "-allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    # Both overlap="compute" prefetches: distinct stages, overlapped, hoisted.
    assert 'tpp.prefetch.stage = 0' in text
    assert 'tpp.prefetch.stage = 1' in text
    assert text.count("tpp.prefetch.overlapped = true") == 2
    assert text.count("tpp.prefetch.hoisted = true") == 2
    # The overlap-policy prefetches hoist above the preceding compute op.
    first_prefetch = text.index("schedule.prefetch")
    matmul = text.index("comp.matmul")
    assert first_prefetch < matmul


@_needs_opt
def test_host_prefetch_recorded_but_not_overlapped(tmp_path):
    # The LSA cold-pool staging space (into="host", overlap="none") must be
    # recorded but carry NO overlap claim — the Gap-2 contract.
    f = tmp_path / "pf.mlir"
    f.write_text(_FIXTURE)
    out = subprocess.run(
        [_OPT, str(f), "-tpp-async-prefetch", "-allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    host_line = next(ln for ln in out.stdout.splitlines() if 'into = "host"' in ln)
    assert "tpp.prefetch.overlapped = false" in host_line
    assert "tpp.prefetch.hoisted = false" in host_line


# ── Gap-2 follow-on: LSA Graph→Schedule prefetch emission ────────────────────

_LSA_PASS_SRC = (REPO_ROOT / "src" / "transforms" / "lib" / "AttentionFamilyPasses.cpp")

_LSA_FIXTURE = '''
func.func @lsa(%q: tensor<2x3x16x16xf32>, %k: tensor<2x3x16x16xf32>, %v: tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32> {
  %0 = "tessera.lookahead_sparse_attention"(%q, %k, %v) {window_size = 6 : i64, block_size = 4 : i64, tau = 64 : i64, threshold = 5.000000e-01 : f64, causal = true} : (tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32>
  return %0 : tensor<2x3x16x16xf32>
}
'''


def test_lsa_prefetch_pass_is_declared_and_registered():
    impl = _LSA_PASS_SRC.read_text()
    assert "tessera-lookahead-sparse-prefetch" in impl
    assert "createLookaheadSparsePrefetchPass" in impl
    passes_cpp = (REPO_ROOT / "src" / "transforms" / "lib" / "Passes.cpp").read_text()
    assert "createLookaheadSparsePrefetchPass" in passes_cpp


@_needs_opt
def test_lsa_emits_prefetch_then_async_overlap_records_it(tmp_path):
    # Graph→Schedule: the LSA op emits schedule.prefetch{into=host} and consumes
    # it; the real async-prefetch pass then records it without claiming overlap.
    f = tmp_path / "lsa.mlir"
    f.write_text(_LSA_FIXTURE)
    out = subprocess.run(
        [_OPT, str(f), "-tessera-lookahead-sparse-prefetch", "-tpp-async-prefetch",
         "-allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    pf = next(ln for ln in text.splitlines() if "schedule.prefetch" in ln)
    assert 'into = "host"' in pf
    assert 'overlap = "none"' in pf
    assert 'tessera.lsa.staging = "host_cold_pool"' in pf
    assert "tpp.prefetch.overlapped = false" in pf  # no overlap claimed
    assert "tpp.prefetch.hoisted = false" in pf
    # The LSA op is rewired to consume the prefetch (it appears after it) and
    # is tagged as having emitted its staging prefetch.
    assert "tessera.lsa.prefetch_emitted = true" in text
    assert text.index("schedule.prefetch") < text.index("lookahead_sparse_attention")
