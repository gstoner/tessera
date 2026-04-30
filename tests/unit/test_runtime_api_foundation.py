from __future__ import annotations

import json

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
)


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
    assert get_last_profile().launch_overhead_ms == 0.0
