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


# ── Sprint 3: front-door value mode + full CPU linalg family ────────────────
#
# These exercise the *canonical front door*: canonical_compile(..., target=
# "apple_cpu", options={"apple_target_ir_mode": "value"}) runs the real
# tessera-lower-to-apple_cpu-full pipeline and captures the emitted value IR —
# no test-only IR injection.  Then runtime.launch dispatches the C ABI symbol
# named in metadata["apple_value_calls"] and checks numerics vs numpy.

def _build_module(op_name, result, arg_specs, result_types, returns, kwargs=None):
    """Construct a single-op GraphIRModule. arg_specs / result_types are
    (mlir_str, shape) tuples; result is comma-joined SSA names."""
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )

    def _t(spec):
        s, shp = spec
        return IRType(s, tuple(str(x) for x in shp), "fp32")

    args = [IRArg(f"a{i}", _t(spec)) for i, spec in enumerate(arg_specs)]
    rtypes = [_t(spec) for spec in result_types]
    rt_mlir = ("(" + ", ".join(t.mlir_str for t in rtypes) + ")"
               if len(rtypes) > 1 else rtypes[0].mlir_str)
    op = IROp(
        result=result, op_name=op_name,
        operands=[f"%a{i}" for i in range(len(arg_specs))],
        operand_types=[spec[0] for spec in arg_specs],
        result_type=rt_mlir, kwargs=kwargs or {},
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=args, result_types=rtypes, body=[op],
        return_values=returns)])


def _front_door(module):
    """Compile a module through the canonical front door in value mode and
    return the RuntimeArtifact."""
    from tessera.compiler.canonical_compile import canonical_compile
    res = canonical_compile(module, target="apple_cpu",
                            options={"apple_target_ir_mode": "value"})
    return res.to_runtime_artifact()


def test_front_door_value_mode_routes_cholesky_without_injected_ir():
    """canonical_compile value mode produces compiler_path == apple_value_target_ir
    and a value IR carrying the cholesky cpu.call — entirely from a Graph IR
    module, no hand-written/injected Target IR."""
    from tessera.compiler import driver
    m = _build_module("tessera.cholesky", "c",
                      [("tensor<3x3xf32>", (3, 3))],
                      [("tensor<3x3xf32>", (3, 3))], ["%c"])
    art = _front_door(m)
    assert art.metadata["compiler_path"] == "apple_value_target_ir"
    assert art.metadata["apple_target_ir_kind"] == "value_target_ir"
    assert art.metadata["executable"] is True
    # The value IR really came from the -full pipeline, not an injected string.
    assert "tessera_apple.cpu.call" in (art.target_ir or "")
    calls = driver.extract_apple_value_calls(art.target_ir or "")
    assert calls and calls[0]["symbol"] == "tessera_apple_cpu_cholesky_f32"


def test_front_door_default_mode_stays_artifact():
    """Without the value option the front door keeps the artifact lane — no
    drift for dashboards / apple_cpu_accelerate."""
    m = _build_module("tessera.cholesky", "c",
                      [("tensor<3x3xf32>", (3, 3))],
                      [("tensor<3x3xf32>", (3, 3))], ["%c"])
    from tessera.compiler.canonical_compile import canonical_compile
    res = canonical_compile(m, target="apple_cpu")
    meta = res.to_runtime_artifact().metadata
    assert meta.get("compiler_path") != "apple_value_target_ir"
    assert meta["apple_target_ir_kind"] != "value_target_ir"


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="Accelerate-LAPACK linalg is Darwin-only")
def test_front_door_launch_cholesky_end_to_end():
    """Front door → launch: no injected IR, dispatched by the IR-named symbol."""
    import numpy as np
    from tessera.runtime import launch
    rng = np.random.default_rng(0)
    mm = rng.standard_normal((4, 4)).astype(np.float32)
    a = (mm @ mm.T + 4 * np.eye(4, dtype=np.float32)).astype(np.float32)
    m = _build_module("tessera.cholesky", "c",
                      [("tensor<4x4xf32>", (4, 4))],
                      [("tensor<4x4xf32>", (4, 4))], ["%c"])
    res = launch(_front_door(m), a)
    assert res["ok"] is True, res
    assert res["compiler_path"] == "apple_value_target_ir"
    np.testing.assert_allclose(res["output"], np.linalg.cholesky(a),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="Accelerate-LAPACK linalg is Darwin-only")
def test_front_door_launch_qr_multi_result_end_to_end():
    """Multi-result op (QR) through the front door returns a tuple in SSA order
    (Q, R) with the right shapes, no injected IR."""
    import numpy as np
    from tessera.runtime import launch
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 4)).astype(np.float32)
    m = _build_module("tessera.qr", "q,r",
                      [("tensor<6x4xf32>", (6, 4))],
                      [("tensor<6x4xf32>", (6, 4)), ("tensor<4x4xf32>", (4, 4))],
                      ["%q", "%r"])
    res = launch(_front_door(m), a)
    assert res["ok"] is True, res
    out = res["output"]
    assert isinstance(out, tuple) and len(out) == 2
    q, r = out
    assert q.shape == (6, 4) and r.shape == (4, 4)
    np.testing.assert_allclose(q @ r, a, rtol=1e-3, atol=1e-3)
    assert np.allclose(np.tril(r, -1), 0.0, atol=1e-4)  # R upper-triangular


# Per-op numerical conformance through the front door + launch (Darwin).
@pytest.mark.skipif(sys.platform != "darwin",
                    reason="Accelerate-LAPACK linalg is Darwin-only")
class TestFrontDoorLinalgFamilyLaunch:
    def _spd(self, n, rng):
        import numpy as np
        m = rng.standard_normal((n, n)).astype(np.float32)
        return (m @ m.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)

    def test_cholesky(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(10)
        a = self._spd(5, rng)
        m = _build_module("tessera.cholesky", "c",
                          [("tensor<5x5xf32>", (5, 5))],
                          [("tensor<5x5xf32>", (5, 5))], ["%c"])
        r = launch(_front_door(m), a)
        assert r["ok"], r
        np.testing.assert_allclose(r["output"], np.linalg.cholesky(a),
                                   rtol=1e-4, atol=1e-4)

    def test_tri_solve(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(11)
        n, k = 5, 3
        L = (np.tril(rng.standard_normal((n, n)).astype(np.float32))
             + n * np.eye(n, dtype=np.float32))
        b = rng.standard_normal((n, k)).astype(np.float32)
        m = _build_module(
            "tessera.tri_solve", "x",
            [("tensor<5x5xf32>", (n, n)), ("tensor<5x3xf32>", (n, k))],
            [("tensor<5x3xf32>", (n, k))], ["%x"],
            kwargs={"lower": True, "trans": False, "unit_diag": False})
        r = launch(_front_door(m), [L, b])  # multi-operand → list args
        assert r["ok"], r
        np.testing.assert_allclose(L @ r["output"], b, rtol=1e-3, atol=1e-3)

    def test_cholesky_solve(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(12)
        n, k = 6, 2
        a = self._spd(n, rng)
        b = rng.standard_normal((n, k)).astype(np.float32)
        m = _build_module(
            "tessera.cholesky_solve", "x",
            [("tensor<6x6xf32>", (n, n)), ("tensor<6x2xf32>", (n, k))],
            [("tensor<6x2xf32>", (n, k))], ["%x"], kwargs={"lower": True})
        r = launch(_front_door(m), [a, b])
        assert r["ok"], r
        np.testing.assert_allclose(a @ r["output"], b, rtol=1e-3, atol=1e-3)

    def test_lu_p_a_eq_l_u(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(13)
        n = 6
        a = self._spd(n, rng)
        m = _build_module("tessera.lu", "lu,piv",
                          [("tensor<6x6xf32>", (n, n))],
                          [("tensor<6x6xf32>", (n, n)), ("tensor<6xi32>", (n,))],
                          ["%lu", "%piv"])
        r = launch(_front_door(m), a)
        assert r["ok"], r
        lu, piv = r["output"]
        assert lu.shape == (n, n) and piv.shape == (n,)
        Lm = np.tril(lu, -1) + np.eye(n, dtype=np.float32)
        Um = np.triu(lu)
        Ap = a.copy()
        for i in range(n):  # apply 1-based sequential row swaps → P*A
            j = int(piv[i]) - 1
            Ap[[i, j]] = Ap[[j, i]]
        np.testing.assert_allclose(Ap, Lm @ Um, rtol=1e-3, atol=1e-3)

    def test_qr(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(14)
        a = rng.standard_normal((5, 3)).astype(np.float32)
        m = _build_module("tessera.qr", "q,r",
                          [("tensor<5x3xf32>", (5, 3))],
                          [("tensor<5x3xf32>", (5, 3)), ("tensor<3x3xf32>", (3, 3))],
                          ["%q", "%r"])
        r = launch(_front_door(m), a)
        assert r["ok"], r
        q, rr = r["output"]
        np.testing.assert_allclose(q @ rr, a, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(q.T @ q, np.eye(3), rtol=1e-3, atol=1e-3)

    def test_svd(self):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(15)
        a = rng.standard_normal((6, 4)).astype(np.float32)
        m = _build_module(
            "tessera.svd", "u,s,v",
            [("tensor<6x4xf32>", (6, 4))],
            [("tensor<6x4xf32>", (6, 4)), ("tensor<4xf32>", (4,)),
             ("tensor<4x4xf32>", (4, 4))], ["%u", "%s", "%v"])
        r = launch(_front_door(m), a)
        assert r["ok"], r
        u, s, v = r["output"]
        assert u.shape == (6, 4) and s.shape == (4,) and v.shape == (4, 4)
        np.testing.assert_allclose(u @ np.diag(s) @ v, a, rtol=1e-3, atol=1e-3)
        assert np.all(s[:-1] >= s[1:] - 1e-4)  # descending singular values


def test_value_mode_output_has_no_husk_or_tile_leftover():
    """The captured value IR from the front door is value-preserving — no
    ub.poison / tensor.empty / tile.* survive."""
    m = _build_module("tessera.svd", "u,s,v",
                      [("tensor<6x4xf32>", (6, 4))],
                      [("tensor<6x4xf32>", (6, 4)), ("tensor<4xf32>", (4,)),
                       ("tensor<4x4xf32>", (4, 4))], ["%u", "%s", "%v"])
    art = _front_door(m)
    ir = art.target_ir or ""
    assert "tessera_apple.cpu.call" in ir
    for bad in ("ub.poison", "tensor.empty", "tile.svd", "tile.qr",
                "tile.lu", "tile.cholesky"):
        assert bad not in ir, f"forbidden '{bad}' survived in value-mode front-door IR"


# ── Sprint 4: resolver hardening + honest coverage boundaries ───────────────
#
# Sprint 4 fixes the Sprint 3 portability blocker: the value front door must
# find the in-repo build/tools/tessera-opt/tessera-opt without TESSERA_OPT or
# PATH (the old parents[2] fallback pointed at python/build/, which never
# exists). It also pins the coverage boundary: CPU linalg value calls execute;
# GPU value calls and non-linalg value calls stay gated / artifact-mode.

_REPO_BUILT_OPT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"


def test_repo_root_walk_finds_source_root():
    """_tessera_repo_root walks up to the dir containing python/tessera AND
    src/compiler — not the python/ dir the old parents[2] assumption hit."""
    from tessera.compiler import driver
    root = driver._tessera_repo_root()
    assert root is not None
    assert (root / "python" / "tessera").is_dir()
    assert (root / "src" / "compiler").is_dir()
    assert root == REPO_ROOT


@pytest.mark.skipif(not _REPO_BUILT_OPT.is_file(),
                    reason="repo-built tessera-opt is absent")
def test_resolver_finds_repo_built_opt_without_env_or_path(monkeypatch, tmp_path):
    """With TESSERA_OPT cleared and PATH scrubbed of tessera-opt, the resolver
    still finds the in-repo build (the Sprint 3 portability fix)."""
    from tessera.compiler import driver
    monkeypatch.delenv("TESSERA_OPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no tessera-opt on this PATH
    assert shutil.which("tessera-opt") is None  # precondition: PATH is scrubbed
    resolved = driver._resolve_tessera_opt()
    assert resolved is not None
    assert Path(resolved) == _REPO_BUILT_OPT


def test_resolver_precedence_env_first(monkeypatch):
    """TESSERA_OPT takes precedence over PATH and the in-repo build."""
    from tessera.compiler import driver
    monkeypatch.setenv("TESSERA_OPT", "/custom/tessera-opt")
    assert driver._resolve_tessera_opt() == "/custom/tessera-opt"


@pytest.mark.skipif(not _REPO_BUILT_OPT.is_file(),
                    reason="repo-built tessera-opt is absent")
def test_front_door_value_mode_works_without_environment(monkeypatch, tmp_path):
    """canonical_compile value mode succeeds with no TESSERA_OPT and a scrubbed
    PATH — proving the front door relies on the resolver, not the environment."""
    from tessera.compiler.canonical_compile import canonical_compile
    monkeypatch.delenv("TESSERA_OPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    m = _build_module("tessera.cholesky", "c",
                      [("tensor<3x3xf32>", (3, 3))],
                      [("tensor<3x3xf32>", (3, 3))], ["%c"])
    art = canonical_compile(m, target="apple_cpu",
                            options={"apple_target_ir_mode": "value"}).to_runtime_artifact()
    assert art.metadata["compiler_path"] == "apple_value_target_ir"
    assert art.metadata["apple_target_ir_kind"] == "value_target_ir"
    assert art.metadata.get("apple_value_target_ir_error") is None


def test_value_mode_failure_records_named_error(monkeypatch):
    """When the -full lowering can't run (bad tessera-opt), the front door keeps
    the artifact IR but records apple_value_target_ir_error — never silent."""
    from tessera.compiler.canonical_compile import canonical_compile
    monkeypatch.setenv("TESSERA_OPT", "/nonexistent/tessera-opt-xyz")
    m = _build_module("tessera.cholesky", "c",
                      [("tensor<3x3xf32>", (3, 3))],
                      [("tensor<3x3xf32>", (3, 3))], ["%c"])
    art = canonical_compile(m, target="apple_cpu",
                            options={"apple_target_ir_mode": "value"}).to_runtime_artifact()
    assert art.metadata.get("compiler_path") != "apple_value_target_ir"
    assert art.metadata["apple_target_ir_kind"] == "target_ir_artifact"
    err = art.metadata.get("apple_value_target_ir_error")
    assert err and "tessera-opt" in err


# GPU linalg value mode: classified but gated (no execution, no fallback output).
@pytest.mark.parametrize("op,result,args,rtype,returns,kwargs", [
    ("tessera.cholesky", "c", [("tensor<8x8xf32>", (8, 8))],
     "tensor<8x8xf32>", ["%c"], None),
    ("tessera.tri_solve", "x",
     [("tensor<4x4xf32>", (4, 4)), ("tensor<4x2xf32>", (4, 2))],
     "tensor<4x2xf32>", ["%x"], {"lower": True, "trans": False, "unit_diag": False}),
])
def test_gpu_value_mode_linalg_is_classified_and_gated(op, result, args, rtype, returns, kwargs):
    """apple_gpu cholesky / tri_solve lower to gpu.kernel_call and route to the
    non-executable apple_gpu/apple_value_target_ir matrix row — structured
    non-success, never fallback output."""
    from tessera.runtime import launch
    m = _build_module(op, result, args, [(rtype, ())], returns, kwargs=kwargs)
    art = _front_door_gpu(m)
    assert art.metadata["apple_target_ir_kind"] == "value_target_ir"
    calls = art.metadata.get("apple_value_calls") or []
    assert calls and calls[0]["op"] == "tessera_apple.gpu.kernel_call"
    # The GPU value row is non-executable: launch must not fabricate output.
    assert art.metadata.get("executable") is not True
    res = launch(art, None)
    assert res["ok"] is False
    assert "output" not in res


def _front_door_gpu(module):
    from tessera.compiler.canonical_compile import canonical_compile
    return canonical_compile(module, target="apple_gpu",
                             options={"apple_target_ir_mode": "value"}).to_runtime_artifact()


# Non-linalg coverage probes — prove current artifact/gated behavior; do NOT
# advertise these as value-executable. Each must satisfy:
#   (a) default mode keeps its prior path (never apple_value_target_ir), and
#   (b) value mode does not become value-executable — either no value-call
#       routing, or a recorded apple_value_target_ir_error (named diagnostic).
#   NOTE (Sprint 5): tessera.matmul is no longer in this list — fp32 rank-2
#   matmul is now an executable CPU value call (see TestSprint5Matmul below).
_NON_LINALG_PROBES = [
    ("tessera.softmax", "o", [("a", "tensor<4x4xf32>", (4, 4))], "tensor<4x4xf32>"),
    ("tessera.gelu", "o", [("a", "tensor<4x4xf32>", (4, 4))], "tensor<4x4xf32>"),
    ("tessera.conv2d", "o",
     [("a", "tensor<1x8x8x3xf32>", (1, 8, 8, 3)),
      ("b", "tensor<3x3x3x4xf32>", (3, 3, 3, 4))], "tensor<1x6x6x4xf32>"),
]


def _build_non_linalg(op_name, result, arg_specs, rtype):
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    args = [IRArg(n, IRType(t, tuple(str(x) for x in shp), "fp32"))
            for (n, t, shp) in arg_specs]
    op = IROp(result=result, op_name=op_name,
              operands=[f"%{n}" for (n, _, _) in arg_specs],
              operand_types=[t for (_, t, _) in arg_specs],
              result_type=rtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=args, result_types=[IRType(rtype, (), "fp32")],
        body=[op], return_values=[f"%{result}"])])


@pytest.mark.parametrize("target", ["apple_cpu", "apple_gpu"])
@pytest.mark.parametrize("op,result,arg_specs,rtype", _NON_LINALG_PROBES)
def test_non_linalg_default_mode_never_value_path(target, op, result, arg_specs, rtype):
    """Default (artifact) compile of matmul/softmax/gelu/conv2d never routes to
    the value lane — Sprint 4 does not add value execution for these."""
    from tessera.compiler.canonical_compile import canonical_compile
    m = _build_non_linalg(op, result, arg_specs, rtype)
    meta = canonical_compile(m, target=target).to_runtime_artifact().metadata
    assert meta.get("compiler_path") != "apple_value_target_ir"
    assert meta.get("apple_target_ir_kind") != "value_target_ir"


@pytest.mark.parametrize("target", ["apple_cpu", "apple_gpu"])
@pytest.mark.parametrize("op,result,arg_specs,rtype", _NON_LINALG_PROBES)
def test_non_linalg_value_mode_not_value_executable(target, op, result, arg_specs, rtype):
    """Value mode for non-linalg ops is honestly bounded: it never produces an
    executable value call. Either no value routing (compiler_path unchanged) or
    a recorded apple_value_target_ir_error — but never a value cpu.call that the
    runtime would dispatch."""
    from tessera.compiler.canonical_compile import canonical_compile
    m = _build_non_linalg(op, result, arg_specs, rtype)
    meta = canonical_compile(m, target=target,
                             options={"apple_target_ir_mode": "value"}).to_runtime_artifact().metadata
    # Never routes through the value executor.
    assert meta.get("compiler_path") != "apple_value_target_ir"
    # No executable cpu.call value op was emitted for a non-linalg op.
    calls = meta.get("apple_value_calls") or []
    assert not any(c.get("op") == "tessera_apple.cpu.call"
                   and c.get("status") == "executable" for c in calls), \
        f"{op} on {target} unexpectedly produced an executable CPU value call"


# ── Sprint 5: first non-linalg executable value op — CPU fp32 rank-2 matmul ──
#
# fp32 rank-2 matmul/gemm is the first non-linalg op to execute on the value
# lane. The TilingPass value path preserves it as a single tile op (no scf.for),
# Tile→Apple emits a tessera_apple.cpu.call with tessera_apple_cpu_gemm_f32, and
# the runtime dispatches Accelerate cblas_sgemm. GPU + non-fp32 + batched stay
# gated.

def _matmul_module(M, K, N, dtype="f32", mlir_dtype="f32"):
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    ta = IRType(f"tensor<{M}x{K}x{mlir_dtype}>", (str(M), str(K)), dtype)
    tb = IRType(f"tensor<{K}x{N}x{mlir_dtype}>", (str(K), str(N)), dtype)
    tc = IRType(f"tensor<{M}x{N}x{mlir_dtype}>", (str(M), str(N)), dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=[IRArg("a", ta), IRArg("b", tb)], result_types=[tc],
        body=[IROp(result="c", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[ta.mlir_str, tb.mlir_str],
                   result_type=tc.mlir_str)],
        return_values=["%c"])])


def test_cpu_full_matmul_lowers_to_gemm_value_call():
    """Static rank-2 f32 matmul lowers (via the value `-full` pipeline) to a
    tessera_apple.cpu.call carrying the tessera_apple_cpu_gemm_f32 symbol — no
    husk, no scf.for, no surviving tile.matmul."""
    body = ('func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) '
            '-> tensor<4x16xf32> {\n'
            '  %0 = tessera.matmul %a, %b : '
            '(tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>\n'
            '  return %0 : tensor<4x16xf32>\n}')
    p = _run("tessera-lower-to-apple_cpu-full", body)
    assert p.returncode == 0, p.stderr
    assert "tessera_apple.cpu.call" in p.stdout
    assert 'symbol = "tessera_apple_cpu_gemm_f32"' in p.stdout
    assert 'op_kind = "matmul"' in p.stdout
    for bad in ("ub.poison", "tensor.empty", "scf.for", "tile.matmul"):
        assert bad not in p.stdout, f"forbidden '{bad}' in matmul value output"


def test_cpu_full_matmul_symbol_on_runtime_allowlist():
    """The gemm symbol the value lowering writes is exactly what the runtime
    value executor dispatches (IR ↔ runtime agreement)."""
    from tessera.compiler import driver
    from tessera.runtime import _APPLE_VALUE_CPU_SYMBOLS
    body = ('func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) '
            '-> tensor<4x16xf32> {\n'
            '  %0 = tessera.matmul %a, %b : '
            '(tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>\n'
            '  return %0 : tensor<4x16xf32>\n}')
    p = _run("tessera-lower-to-apple_cpu-full", body)
    assert p.returncode == 0, p.stderr
    calls = driver.extract_apple_value_calls(p.stdout)
    assert calls and calls[0]["symbol"] == "tessera_apple_cpu_gemm_f32"
    assert calls[0]["symbol"] in _APPLE_VALUE_CPU_SYMBOLS


def test_front_door_matmul_routes_value_without_injection():
    """canonical_compile value mode routes a Graph IR matmul to the value lane
    (compiler_path == apple_value_target_ir, gemm symbol), no injected IR."""
    art = _front_door(_matmul_module(4, 8, 16))
    assert art.metadata["compiler_path"] == "apple_value_target_ir"
    assert art.metadata["apple_target_ir_kind"] == "value_target_ir"
    assert "tessera_apple_cpu_gemm_f32" in (art.target_ir or "")
    calls = art.metadata.get("apple_value_calls") or []
    assert calls and calls[0]["symbol"] == "tessera_apple_cpu_gemm_f32"


def test_front_door_default_matmul_stays_artifact():
    """Default matmul compile keeps the artifact path (no value routing)."""
    from tessera.compiler.canonical_compile import canonical_compile
    meta = canonical_compile(_matmul_module(4, 8, 16),
                             target="apple_cpu").to_runtime_artifact().metadata
    assert meta.get("compiler_path") != "apple_value_target_ir"
    assert meta.get("apple_target_ir_kind") != "value_target_ir"


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="Accelerate cblas_sgemm is Darwin-only")
class TestSprint5Matmul:
    def _launch(self, M, K, N, seed):
        import numpy as np
        from tessera.runtime import launch
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        r = launch(_front_door(_matmul_module(M, K, N)), [a, b])
        return r, a, b

    def test_matmul_matches_numpy(self):
        import numpy as np
        for (M, K, N), seed in (((4, 8, 16), 0), ((32, 32, 32), 1),
                                ((1, 7, 5), 2), ((17, 3, 9), 3)):
            r, a, b = self._launch(M, K, N, seed)
            assert r["ok"], r
            assert r["compiler_path"] == "apple_value_target_ir"
            np.testing.assert_allclose(r["output"], a @ b, rtol=1e-4, atol=1e-4)

    def test_matmul_output_shape(self):
        r, a, b = self._launch(6, 10, 4, 7)
        assert r["ok"], r
        assert r["output"].shape == (6, 4)


# ── Sprint 5 boundaries: GPU + non-fp32 + batched matmul stay gated ─────────

def test_gpu_value_matmul_is_gated_no_output():
    """apple_gpu value matmul never becomes value-executable and launch returns
    a structured non-success (no fabricated output)."""
    from tessera.runtime import launch
    art = _front_door_gpu(_matmul_module(4, 8, 16))
    # GPU `-full` does not preserve matmul as a value op, so it is not routed to
    # the executable value lane.
    assert art.metadata.get("compiler_path") != "apple_value_target_ir" \
        or art.metadata.get("executable") is not True
    calls = art.metadata.get("apple_value_calls") or []
    assert not any(c.get("op") == "tessera_apple.cpu.call"
                   and c.get("status") == "executable" for c in calls)
    res = launch(art, None)
    assert res["ok"] is False
    assert "output" not in res


@pytest.mark.parametrize("mlir_dtype,dtype", [("f16", "fp16"), ("bf16", "bf16")])
def test_non_fp32_matmul_value_mode_is_gated(mlir_dtype, dtype):
    """f16/bf16 matmul is outside the Sprint 5 envelope and is gated with a
    *named* unsupported reason — never a silent f32 fallback. The gate can fire
    at either layer: the Graph IR target-capability verifier (which rejects
    non-fp32 matmul on CPU outright) or, if it ever reaches lowering, a recorded
    apple_value_target_ir_error. Both are honest; neither is value-executable."""
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.compiler.graph_ir import GraphIRVerificationError
    m = _matmul_module(4, 8, 16, dtype=dtype, mlir_dtype=mlir_dtype)
    try:
        meta = canonical_compile(m, target="apple_cpu",
                                 options={"apple_target_ir_mode": "value"}).to_runtime_artifact().metadata
    except GraphIRVerificationError as exc:
        # Earliest honest gate: the target-capability verifier names the dtype.
        assert dtype in str(exc) or "not supported" in str(exc)
        return
    # If it compiled at all, it must NOT be value-executable.
    assert meta.get("compiler_path") != "apple_value_target_ir"
    err = meta.get("apple_value_target_ir_error")
    assert err and ("matmul" in err or "envelope" in err), err


def test_batched_matmul_value_mode_is_gated():
    """Batched (rank-3) matmul is outside the envelope — value mode is gated
    with a recorded reason, never an executable f32 gemm call."""
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    ta = IRType("tensor<2x4x8xf32>", ("2", "4", "8"), "fp32")
    tb = IRType("tensor<2x8x16xf32>", ("2", "8", "16"), "fp32")
    tc = IRType("tensor<2x4x16xf32>", ("2", "4", "16"), "fp32")
    m = GraphIRModule(functions=[GraphIRFunction(
        name="f", args=[IRArg("a", ta), IRArg("b", tb)], result_types=[tc],
        body=[IROp(result="c", op_name="tessera.batched_gemm",
                   operands=["%a", "%b"],
                   operand_types=[ta.mlir_str, tb.mlir_str],
                   result_type=tc.mlir_str)],
        return_values=["%c"])])
    meta = canonical_compile(m, target="apple_cpu",
                             options={"apple_target_ir_mode": "value"}).to_runtime_artifact().metadata
    assert meta.get("compiler_path") != "apple_value_target_ir"
    calls = meta.get("apple_value_calls") or []
    assert not any(c.get("op") == "tessera_apple.cpu.call"
                   and c.get("status") == "executable" for c in calls)


# ── Sprint 5 review fixes (P1/P2) ───────────────────────────────────────────

def test_matmul_result_shape_mismatch_is_rejected():
    """P1: a malformed (4x8)@(8x16)->(5x5) passes rank+static+f32+K but must NOT
    become an executable value call. The MatmulOp verifier rejects it with a
    named result-dimension diagnostic before any value lowering."""
    body = ('func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) '
            '-> tensor<5x5xf32> {\n'
            '  %0 = tessera.matmul %a, %b : '
            '(tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<5x5xf32>\n'
            '  return %0 : tensor<5x5xf32>\n}')
    p = _run("tessera-lower-to-apple_cpu-full", body)
    assert p.returncode != 0
    assert "tessera_apple.cpu.call" not in p.stdout
    assert "result row dimension" in p.stderr or "result column dimension" in p.stderr


def test_matmul_value_executor_requires_exact_operand_count():
    """P2: the CPU value executor requires an exact operand count — an extra
    operand is rejected (invalid_artifact), never silently ignored."""
    from tessera.runtime import RuntimeArtifact, launch
    import numpy as np
    art = RuntimeArtifact(metadata={
        "target": "apple_cpu",
        "compiler_path": "apple_value_target_ir",
        "executable": True,
        "apple_value_calls": [{
            "op": "tessera_apple.cpu.call", "op_kind": "matmul",
            "symbol": "tessera_apple_cpu_gemm_f32", "status": "executable",
        }],
    })
    a = np.ones((4, 8), np.float32)
    b = np.ones((8, 16), np.float32)
    extra = np.ones((1, 1), np.float32)
    res = launch(art, [a, b, extra])  # one operand too many
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "exactly 2" in res["reason"] or "input" in res["reason"]
