from __future__ import annotations

import json

import numpy as np

import tessera
from tessera.runtime import (
    RuntimeArtifact,
    available_backends,
    backend_capabilities,
    compile,
    get_last_profile,
    launch,
    load_artifact,
    query_backend,
    runtime_smoke_telemetry,
)
from tessera.testing import compile_and_maybe_launch


def test_runtime_backend_query_reports_cpu():
    backends = available_backends()
    assert "cpu" in backends
    cap = backend_capabilities("cpu")
    assert cap.available is True
    assert query_backend("cpu")["name"] == "cpu"


def test_runtime_artifact_round_trip_and_hash_stable():
    artifact = RuntimeArtifact(graph_ir="module { tessera.gemm }", metadata={"target": "cpu"})
    encoded = artifact.to_json()
    decoded = load_artifact(encoded)
    assert decoded.graph_ir == artifact.graph_ir
    assert decoded.artifact_hash == artifact.artifact_hash


def test_compile_artifact_helper_is_exported():
    artifact = tessera.compile_artifact("module { tessera.fft }", target="cpu")
    assert isinstance(artifact, RuntimeArtifact)
    assert artifact.metadata["target"] == "cpu"


def test_launch_artifact_reports_unsupported_not_success():
    artifact = compile("module { tessera.gemm }", target="cpu")
    result = launch(artifact, args=[])
    assert result["ok"] is False
    assert result["compiler_path"] == "artifact_only"
    assert result["runtime_status"] in {"unsupported", "missing_backend"}
    assert result["telemetry"]["schema"] == "tessera.telemetry.v1"
    assert result["telemetry"]["status"] in {"unsupported", "missing_backend"}
    assert get_last_profile().launch_overhead_ms == 0.0


def test_jit_cpu_runtime_artifact_launches_successfully():
    @tessera.jit
    def mm(A, B):
        return tessera.ops.matmul(A, B)

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)

    artifact = mm.runtime_artifact()
    assert isinstance(artifact, RuntimeArtifact)
    assert artifact.metadata["executable"] is True
    assert artifact.metadata["compiler_path"] == "jit_cpu_numpy"
    assert artifact.metadata["input_descriptors"] == [{"name": "A"}, {"name": "B"}]
    assert artifact.metadata["output_descriptor"]["name"] == artifact.metadata["output_name"]
    assert artifact.schedule_ir and artifact.tile_ir and artifact.target_ir

    loaded = load_artifact(artifact.to_json())
    result = launch(loaded, args=(a, b))

    assert result["ok"] is True
    assert result["runtime_status"] == "success"
    assert result["compiler_path"] == "jit_cpu_numpy"
    assert result["telemetry"]["source"] == "runtime"
    assert result["telemetry"]["status"] == "ok"
    np.testing.assert_allclose(result["output"], a @ b)
    assert get_last_profile().cpu_wall_ms is not None


def test_jit_cpu_runtime_artifact_launch_supports_dict_args_and_stable_hash():
    @tessera.jit
    def stable_probs(x):
        return tessera.ops.softmax(tessera.ops.relu(x), axis=0)

    x = np.array([-1.0, 2.0], dtype=np.float32)
    artifact = stable_probs.runtime_artifact()
    assert artifact.artifact_hash == load_artifact(artifact.to_json()).artifact_hash

    result = launch(artifact, args={"x": x})

    assert result["ok"] is True
    np.testing.assert_allclose(result["output"], tessera.ops.softmax(tessera.ops.relu(x), axis=0))


def test_non_executable_runtime_artifact_keeps_structured_status():
    @tessera.jit
    def unsupported(x):
        return tessera.ops.dropout(x)

    artifact = unsupported.runtime_artifact()
    result = launch(artifact, args=(np.ones(2, dtype=np.float32),))

    assert artifact.metadata["executable"] is False
    assert result["ok"] is False
    assert result["runtime_status"] == "unsupported"
    assert result["compiler_path"] == "eager_fallback"
    assert result["telemetry"]["bottleneck"] == "failed_or_unmeasured"


def test_runtime_smoke_telemetry_exercises_cpu_spine():
    payload = runtime_smoke_telemetry(mock=True, bytes_size=32)

    assert payload["schema"] == "tessera.telemetry.v1"
    assert payload["runtime_status"] == "success"
    assert payload["device"]["kind"] == "CPU"
    assert payload["mapped_bytes"] == 32
    assert payload["event_timestamp_ns"] > 0
    names = {event["name"] for event in payload["telemetry_events"]}
    assert {"runtime.init", "runtime.malloc", "runtime.record_event", "runtime.shutdown"} <= names


def test_compiler_harness_captures_artifact_launch_and_diagnostics():
    def relu_fn(x):
        return tessera.ops.relu(x)

    x = np.array([-1.0, 2.0], dtype=np.float32)
    result = compile_and_maybe_launch(relu_fn, x)

    assert result.artifact.metadata["compiler_path"] == "jit_cpu_numpy"
    assert result.launch_result is not None
    assert result.launch_result["ok"] is True
    assert any("JIT_COMPILED_CPU" in item for item in result.diagnostics)
    np.testing.assert_allclose(result.launch_result["output"], tessera.ops.relu(x))
