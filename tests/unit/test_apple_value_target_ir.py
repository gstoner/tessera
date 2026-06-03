"""Apple Value Target IR sprint — pipeline-intent contract guard.

The `-full` pipelines are *value-preserving*: they emit value-producing
tessera_apple.{cpu.call,gpu.kernel_call} ops that carry real SSA results, and
the final module must contain NO `ub.poison`, NO `tensor.empty`, and NO
surviving `tile.*`.  The artifact pipelines remain metadata-only and may use the
`ub.poison` husk.  This guard pins both sides via the real `tessera-opt`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_OPT_DEFAULT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _find_opt() -> str | None:
    if _OPT_DEFAULT.is_file() and os.access(_OPT_DEFAULT, os.X_OK):
        return str(_OPT_DEFAULT)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
pytestmark = pytest.mark.skipif(_OPT is None, reason="tessera-opt not built")

_GRAPH = {
    "cholesky": 'func.func @f(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {\n'
                '  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>\n'
                '  return %0 : tensor<8x8xf32>\n}',
    "tri_solve": 'func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {\n'
                 '  %0 = tessera.tri_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>\n'
                 '  return %0 : tensor<4x2xf32>\n}',
    "svd": 'func.func @f(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {\n'
           '  %u, %s, %v = tessera.svd %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)\n'
           '  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>\n}',
}


def _run(pipeline: str, body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_OPT, f"-{pipeline}", "--allow-unregistered-dialect", "-"],
        input=body, capture_output=True, text=True, timeout=60)


_FORBIDDEN = ("ub.poison", "tensor.empty", "tile.cholesky", "tile.tri_solve",
              "tile.svd", "= tessera.")


def test_cpu_full_is_value_preserving():
    """All 6 CPU linalg ops lower to value cpu.call with no husk / no tile leftover."""
    for op in ("cholesky", "tri_solve", "svd"):
        p = _run("tessera-lower-to-apple_cpu-full", _GRAPH[op])
        assert p.returncode == 0, p.stderr
        assert "tessera_apple.cpu.call" in p.stdout, op
        for bad in _FORBIDDEN:
            assert bad not in p.stdout, f"{op}: forbidden '{bad}' in value-mode output"


def test_gpu_full_value_executable_ops():
    """cholesky + tri_solve lower to gpu.kernel_call (executable); no husk/tile."""
    for op in ("cholesky", "tri_solve"):
        p = _run("tessera-lower-to-apple_gpu-full", _GRAPH[op])
        assert p.returncode == 0, p.stderr
        assert "tessera_apple.gpu.kernel_call" in p.stdout, op
        assert 'status = "executable"' in p.stdout, op
        for bad in _FORBIDDEN:
            assert bad not in p.stdout, f"{op}: forbidden '{bad}' in value-mode output"


def test_gpu_full_nonexecutable_linalg_fails_with_named_diagnostic():
    """svd has no executable GPU dispatch yet → the value-only full pipeline must
    fail loudly with a named diagnostic, never silently degrade to an artifact."""
    p = _run("tessera-lower-to-apple_gpu-full", _GRAPH["svd"])
    assert p.returncode != 0
    assert "no executable GPU dispatch" in p.stderr
    assert "tessera.svd" in p.stderr


def test_front_door_classifies_value_vs_artifact_and_extracts_dispatch():
    """driver.classify_apple_target_ir distinguishes the value lane from the
    artifact lane, and extract_apple_value_calls reads the dispatch tuple the
    runtime consumes (symbol + executable gating)."""
    from tessera.compiler import driver

    # Value lane: -full → value ops → value_target_ir; calls are executable.
    p = _run("tessera-lower-to-apple_cpu-full", _GRAPH["cholesky"])
    assert p.returncode == 0, p.stderr
    assert driver.classify_apple_target_ir(p.stdout) == "value_target_ir"
    calls = driver.extract_apple_value_calls(p.stdout)
    assert len(calls) == 1
    c = calls[0]
    assert c["op"] == "tessera_apple.cpu.call"
    assert c["op_kind"] == "cholesky"
    assert c["symbol"] == "tessera_apple_cpu_cholesky_f32"
    assert driver.apple_value_call_is_executable(c)

    # Artifact lane (bare tile → artifact pass) → target_ir_artifact, no value calls.
    body = ('func.func @f(%a: tensor<6x4xf32>) -> tensor<6x4xf32> {\n'
            '  %0 = "tile.svd"(%a) {source = "tessera.svd", result = "v0", ordinal = 0 : i64}'
            ' : (tensor<6x4xf32>) -> tensor<6x4xf32>\n'
            '  return %0 : tensor<6x4xf32>\n}')
    pa = _run("tessera-lower-to-apple_cpu", body)
    assert pa.returncode == 0, pa.stderr
    assert driver.classify_apple_target_ir(pa.stdout) == "target_ir_artifact"
    assert driver.extract_apple_value_calls(pa.stdout) == []


def test_artifact_pipeline_still_emits_metadata_ops():
    """The artifact pipeline keeps emitting value-less metadata ops (consumed by
    dashboards) — the bare Tile op husk path is unchanged."""
    body = ('func.func @f(%a: tensor<6x4xf32>) -> tensor<6x4xf32> {\n'
            '  %0 = "tile.svd"(%a) {source = "tessera.svd", result = "v0", ordinal = 0 : i64}'
            ' : (tensor<6x4xf32>) -> tensor<6x4xf32>\n'
            '  return %0 : tensor<6x4xf32>\n}')
    p = _run("tessera-lower-to-apple_cpu", body)
    assert p.returncode == 0, p.stderr
    assert "tessera_apple.cpu.vector_op" in p.stdout
    assert "ub.poison" in p.stdout  # artifact husk is allowed here
    assert "tessera_apple.cpu.call" not in p.stdout  # value op is NOT in artifact mode
