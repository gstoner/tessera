from __future__ import annotations

import json

import numpy as np

import tessera as ts
from tessera.compiler import CompileRequest, CompileTraceEvent
from tessera.compiler.driver import PIPELINE_BY_TARGET
from tessera.compiler.matmul_pipeline import normalize_target_kind
from tessera.testing.compiler_examples import COMPILER_EXAMPLE_MANIFEST, FOUNDATION_TARGETS, qualify_compiler_example


def test_compile_request_preserves_exact_x86_and_selects_pipeline():
    request = CompileRequest(
        source_origin="unit",
        function_name="mm",
        graph_ir='module { "tessera.matmul"() : () -> () }',
        target="x86",
    )

    # Exact x86 is a native target, not an alias for the CPU reference lane.
    assert request.target == "x86"
    assert request.pipeline_name == "tessera-lower-to-x86"
    assert request.graph_hash
    assert request.to_dict()["graph_ir_hash"] == request.graph_hash
    assert normalize_target_kind("x86_64") == "cpu"


def test_spine_foundation_preserves_existing_driver_pipeline_selection():
    expected = {
        "apple_cpu": "tessera-lower-to-apple_cpu",
        "apple_gpu": "tessera-lower-to-apple_gpu-runtime",
        "cpu": "tessera-lower-to-x86",
        "nvidia_sm100": "tessera-lower-to-gpu",
        "nvidia_sm120": "tessera-lower-to-gpu",
        "nvidia_sm80": "tessera-lower-to-gpu",
        "nvidia_sm90": "tessera-lower-to-gpu",
        "rocm": "tessera-lower-to-rocm",
        "rocm_gfx1100": "tessera-target-artifact",
        "rocm_gfx1151": "tessera-target-artifact",
        "rocm_gfx1200": "tessera-target-artifact",
        "rocm_gfx1201": "tessera-target-artifact",
        "rocm_gfx1250": "tessera-target-artifact",
        "rocm_gfx90a": "tessera-target-artifact",
        "rocm_gfx940": "tessera-target-artifact",
        "rocm_gfx942": "tessera-target-artifact",
        "rocm_gfx950": "tessera-target-artifact",
        "x86": "tessera-lower-to-x86",
    }
    assert PIPELINE_BY_TARGET == expected
    for target, pipeline in expected.items():
        request = CompileRequest(
            source_origin="unit",
            function_name="mm",
            graph_ir='module { "tessera.matmul"() : () -> () }',
            target=target,
        )
        assert request.pipeline_name == pipeline


def test_compile_trace_event_serializes_json_and_chrome_trace():
    event = CompileTraceEvent(
        pass_name="tessera-lower-to-x86",
        target="cpu",
        input_hash="a",
        output_hash="b",
        elapsed_ms=1.25,
        status="ok",
        diagnostic_count=0,
    )

    payload = event.to_dict()
    chrome = event.to_chrome_trace_event()
    assert payload["schema"] == "tessera.compiler.trace.v1"
    assert payload["pass_name"] == "tessera-lower-to-x86"
    assert chrome["cat"] == "tessera.compiler"
    assert chrome["args"]["output_hash"] == "b"


def test_jit_routes_artifacts_and_trace_through_compile_bundle():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(2, dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    np.testing.assert_allclose(mm(A, B), A @ B)

    artifact = mm.runtime_artifact()
    metadata = artifact.metadata
    assert mm.compile_bundle is not None
    assert metadata["pipeline_name"] == PIPELINE_BY_TARGET["cpu"]
    assert metadata["artifact_hashes"]["graph"]
    assert metadata["artifact_hashes"]["target"]
    assert metadata["profiling"]["graph_hash"] == mm.compile_bundle.request.graph_hash
    assert metadata["profiling"]["schedule_hash"] == metadata["artifact_hashes"]["schedule"]
    assert metadata["profiling"]["target_hash"] == metadata["artifact_hashes"]["target"]
    assert metadata["runtime_status"] == "ready"
    assert metadata["executable"] is True
    assert metadata["compiler_path"] == "jit_cpu_numpy"
    assert mm.lowering_trace()[0]["pass_name"] == "python-frontend-artifact-builder"
    assert json.loads(mm.compile_bundle.trace_json())[0]["target"] == "cpu"
    assert "traceEvents" in json.loads(mm.compile_bundle.chrome_trace_json())


def test_debug_env_vars_dump_compiler_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERA_DEBUG_IR", "1")
    monkeypatch.setenv("TESSERA_DUMP_STATE", "1")
    monkeypatch.setenv("TESSERA_DUMP_DIR", str(tmp_path))

    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    mm.runtime_artifact()

    dump_dirs = list(tmp_path.iterdir())
    assert len(dump_dirs) == 1
    dump_dir = dump_dirs[0]
    assert (dump_dir / "graph.mlir").exists()
    assert (dump_dir / "schedule.mlir").exists()
    assert (dump_dir / "tile.mlir").exists()
    assert (dump_dir / "target.mlir").exists()
    metadata = json.loads((dump_dir / "metadata.json").read_text())
    assert metadata["profiling"]["graph_hash"]
    assert "traceEvents" in json.loads((dump_dir / "trace.json").read_text())


def test_non_cpu_target_artifacts_are_traceable_and_non_executable():
    @ts.jit(target="cuda")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    artifact = mm.runtime_artifact()
    metadata = artifact.metadata
    assert metadata["target"] == "nvidia_sm90"
    assert metadata["pipeline_name"] == "tessera-lower-to-gpu"
    assert metadata["runtime_status"] == "artifact_only"
    assert metadata["executable"] is False
    assert metadata["compiler_path"] == "target_ir_artifact"
    assert "tessera_nvidia.wgmma" in artifact.target_ir
    assert any(event["status"] in {"tool-missing", "tool-ok", "tool-failed", "tool-error"} for event in mm.lowering_trace())


def test_compiler_example_manifest_qualifies_each_foundation_target():
    seen = set()
    for example in COMPILER_EXAMPLE_MANIFEST:
        assert set(example.stages_by_target) == set(FOUNDATION_TARGETS)
        for target in FOUNDATION_TARGETS:
            # Runtime execution is claimed only when this compiler artifact is
            # joined to an executable path for the same exact target. Native
            # x86 kernel fixtures do not make every generic program executable.
            claimed = set(example.stages_by_target[target])
            should_run = "runtime-executable" in claimed
            result = qualify_compiler_example(example, target, run=should_run)
            seen.add((example.example_id, result.target))
            assert result.artifact.metadata["artifact_hashes"]["graph"]
            assert result.trace
            if should_run:
                assert result.launch_result is not None
                assert result.launch_result["ok"] is True
                assert result.artifact.metadata["runtime_status"] == "ready"
            elif target == "apple_cpu" and result.artifact.metadata["runtime_status"] == "ready":
                # Manifest examples are matmul-driven and the multi-op
                # runtime path means they now report runtime_status="ready".
                assert result.artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
            else:
                assert result.launch_result is None
                assert result.artifact.metadata["runtime_status"] in {"ready", "artifact_only"}
                if target in {"cuda", "rocm"}:
                    assert result.artifact.metadata["runtime_status"] == "artifact_only"

    assert ("mlp_matmul_relu", "x86") in seen
    assert ("flash_attn_contract", "nvidia_sm90") in seen
