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
import sys
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
    "cholesky_solve": 'func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {\n'
                      '  %0 = tessera.cholesky_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>\n'
                      '  return %0 : tensor<4x2xf32>\n}',
    "lu": 'func.func @f(%a: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>) {\n'
          '  %lu, %p = tessera.lu %a : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>)\n'
          '  return %lu, %p : tensor<4x4xf32>, tensor<4xi32>\n}',
    "qr": 'func.func @f(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x4xf32>) {\n'
          '  %q, %r = tessera.qr %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x4xf32>)\n'
          '  return %q, %r : tensor<6x4xf32>, tensor<4x4xf32>\n}',
    "svd": 'func.func @f(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {\n'
           '  %u, %s, %v = tessera.svd %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)\n'
           '  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>\n}',
}

# Every CPU-converted linalg op (LF1–LF5 + cholesky pilot) lowers via cpu.call.
_CPU_LINALG_OPS = ("cholesky", "tri_solve", "cholesky_solve", "lu", "qr", "svd")


def _run(pipeline: str, body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_OPT, f"-{pipeline}", "--allow-unregistered-dialect", "-"],
        input=body, capture_output=True, text=True, timeout=60)


# The value-mode module must contain none of: the poison/empty husks, any
# surviving tile.<linalg> op, or any leftover Graph-IR tessera.* op.
_FORBIDDEN = (
    "ub.poison", "tensor.empty",
    "tile.cholesky", "tile.tri_solve", "tile.cholesky_solve",
    "tile.lu", "tile.qr", "tile.svd",
    "= tessera.",
)


def test_cpu_full_is_value_preserving():
    """All 6 CPU linalg ops lower to value cpu.call with no husk / no tile leftover."""
    for op in _CPU_LINALG_OPS:
        p = _run("tessera-lower-to-apple_cpu-full", _GRAPH[op])
        assert p.returncode == 0, f"{op}: {p.stderr}"
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


def test_front_door_records_value_target_ir_in_runtime_metadata():
    """RV-P1: the front door (CompileResult.to_runtime_artifact) actually
    *consumes* the classifier/extractor — the runtime artifact metadata records
    apple_target_ir_kind and, for the value lane, the dispatch tuples."""
    import dataclasses

    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.compiler.driver import LoweringArtifact
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )

    t = IRType("tensor<8x8xf32>", ("8", "8"), "fp32")
    fn = GraphIRFunction(
        name="f", args=[IRArg("a", t)], result_types=[t],
        body=[IROp(result="c", op_name="tessera.cholesky", operands=["%a"],
                   operand_types=["tensor<8x8xf32>"], result_type="tensor<8x8xf32>")],
        return_values=["%c"],
    )
    res = canonical_compile(GraphIRModule(functions=[fn]), target="apple_cpu")

    # The key is always recorded for Apple targets (consumption proof).
    base_meta = res.to_runtime_artifact().metadata
    assert "apple_target_ir_kind" in base_meta

    # Inject value-IR as the target IR and confirm the front door classifies it
    # as the value lane and extracts the dispatch tuple the runtime reads.
    value_ir = (
        "module {\n  func.func @f(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {\n"
        '    %0 = tessera_apple.cpu.call %a {op_kind = "cholesky", '
        'symbol = "tessera_apple_cpu_cholesky_f32", abi = "lapack_spotrf", '
        'status = "executable", framework = "Accelerate"} '
        ": (tensor<8x8xf32>) -> tensor<8x8xf32>\n"
        "    return %0 : tensor<8x8xf32>\n  }\n}\n"
    )
    value_bundle = dataclasses.replace(
        res.bundle, target_ir=LoweringArtifact("target", value_ir))
    value_res = dataclasses.replace(res, bundle=value_bundle)
    meta = value_res.to_runtime_artifact().metadata
    assert meta["apple_target_ir_kind"] == "value_target_ir"
    assert meta["apple_value_calls"][0]["symbol"] == "tessera_apple_cpu_cholesky_f32"
    assert meta["apple_value_calls"][0]["status"] == "executable"


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


def test_extractor_survives_line_wrapped_attr_dicts():
    """RV-P3: the extractor anchors to the op mnemonic and spans (DOTALL) to its
    first {...}, so a value op whose attr dict is pretty-printed across multiple
    lines is still parsed correctly."""
    from tessera.compiler import driver

    wrapped = (
        "module {\n"
        "  func.func @f(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {\n"
        "    %0 = tessera_apple.cpu.call %a {\n"
        '        op_kind = "cholesky",\n'
        '        symbol = "tessera_apple_cpu_cholesky_f32",\n'
        '        abi = "lapack_spotrf",\n'
        '        status = "executable",\n'
        '        framework = "Accelerate"\n'
        "    } : (tensor<8x8xf32>) -> tensor<8x8xf32>\n"
        "    return %0 : tensor<8x8xf32>\n"
        "  }\n}\n"
    )
    calls = driver.extract_apple_value_calls(wrapped)
    assert len(calls) == 1
    c = calls[0]
    assert c["op_kind"] == "cholesky"
    assert c["symbol"] == "tessera_apple_cpu_cholesky_f32"
    assert driver.apple_value_call_is_executable(c)


def test_extractor_survives_json_braces_in_argument_layout():
    """RV-P3/S2-1: an argument_layout value that is itself a JSON object (with
    nested braces inside the quoted string) does not terminate the attr dict."""
    from tessera.compiler import driver

    ir = (
        '%0 = tessera_apple.gpu.package_call %a {op_kind = "svd", '
        'symbol = "tessera_apple_gpu_svd_f32", status = "executable", '
        'argument_layout = "{\\"buffers\\":[{\\"idx\\":0}],\\"grid\\":{\\"x\\":1}}"} '
        ": (tensor<6x4xf32>) -> tensor<6x4xf32>"
    )
    calls = driver.extract_apple_value_calls(ir)
    assert len(calls) == 1
    c = calls[0]
    assert c["op_kind"] == "svd" and c["symbol"] == "tessera_apple_gpu_svd_f32"
    assert c["argument_layout"].startswith("{") and "buffers" in c["argument_layout"]


# ── Sprint 2: metadata → execution closure ──────────────────────────────────

def _value_cpu_cholesky_artifact():
    """A RuntimeArtifact whose metadata carries an executable CPU cholesky
    value call (the shape canonical_compile produces for the value lane)."""
    from tessera.runtime import RuntimeArtifact
    return RuntimeArtifact(
        target_ir=(
            '%0 = tessera_apple.cpu.call %a {op_kind = "cholesky", '
            'symbol = "tessera_apple_cpu_cholesky_f32", abi = "lapack_spotrf", '
            'status = "executable", framework = "Accelerate"} '
            ": (tensor<3x3xf32>) -> tensor<3x3xf32>"),
        metadata={
            "target": "apple_cpu",
            "compiler_path": "apple_value_target_ir",
            "executable": True,
            "apple_target_ir_kind": "value_target_ir",
            "apple_value_calls": [{
                "op": "tessera_apple.cpu.call", "op_kind": "cholesky",
                "symbol": "tessera_apple_cpu_cholesky_f32", "status": "executable",
            }],
            "arg_names": ["a"],
        },
    )


def test_value_artifact_routes_through_apple_value_target_ir():
    """The (apple_cpu, apple_value_target_ir) pair resolves to the executable
    value-call row (not the accelerate-gemm row)."""
    from tessera.compiler import execution_matrix as EM
    art = _value_cpu_cholesky_artifact()
    row = EM.executor_for_metadata(art.metadata)
    assert row is not None and row.executable
    assert row.executor_id == "apple_value_target_ir"
    assert row.execution_kind == "native_cpu"


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="Accelerate-LAPACK cholesky is Darwin-only")
def test_launch_executes_cholesky_by_ir_named_symbol():
    """End-to-end: runtime.launch dispatches the symbol named in
    apple_value_calls and returns the correct Cholesky factor."""
    import numpy as np
    from tessera.runtime import launch

    rng = np.random.default_rng(0)
    m = rng.standard_normal((3, 3)).astype(np.float32)
    a = (m @ m.T + 3 * np.eye(3, dtype=np.float32)).astype(np.float32)
    res = launch(_value_cpu_cholesky_artifact(), a)
    assert res["ok"] is True, res
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "apple_value_target_ir"
    np.testing.assert_allclose(res["output"], np.linalg.cholesky(a),
                               rtol=1e-4, atol=1e-4)


def test_gpu_value_call_is_gated_not_executed():
    """An apple_gpu value-target artifact is classified but NOT dispatched —
    launch returns a structured non-success, never a silent fallback."""
    from tessera.runtime import RuntimeArtifact, launch
    art = RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.kernel_call", "op_kind": "cholesky",
            "symbol": "tessera_apple_gpu_cholesky_f32", "status": "executable",
        }],
    })
    res = launch(art, None)
    assert res["ok"] is False
    assert "output" not in res


def test_package_call_is_not_executable_cpu_dispatch():
    """A package_call (or any non-cpu.call) routed to the CPU value executor is
    a named follow-on → invalid_artifact, not a wrong dispatch."""
    from tessera.runtime import RuntimeArtifact, launch
    art = RuntimeArtifact(metadata={
        "target": "apple_cpu",
        "compiler_path": "apple_value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.gpu.package_call", "op_kind": "svd",
            "symbol": "tessera_apple_gpu_svd_f32", "status": "executable",
        }],
        "arg_names": ["a"],
    })
    res = launch(art, None)
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "package_call" in res["reason"] or "cpu.call" in res["reason"]


def test_cpu_full_spine_symbol_matches_runtime_allowlist():
    """The symbol the CPU full-spine lowering writes into the IR is exactly the
    one the runtime value executor will dispatch (IR ↔ runtime agreement)."""
    from tessera.compiler import driver
    from tessera.runtime import _APPLE_VALUE_CPU_SYMBOLS
    body = ('func.func @f(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {\n'
            '  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>\n'
            '  return %0 : tensor<8x8xf32>\n}')
    p = _run("tessera-lower-to-apple_cpu-full", body)
    assert p.returncode == 0, p.stderr
    calls = driver.extract_apple_value_calls(p.stdout)
    assert calls and calls[0]["symbol"] in _APPLE_VALUE_CPU_SYMBOLS


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
