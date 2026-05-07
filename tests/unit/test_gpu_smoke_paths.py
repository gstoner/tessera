from __future__ import annotations

from tessera.compiler.gpu_smoke import run_matmul_smoke


def test_nvidia_smoke_is_hardware_gated_and_structured():
    result = run_matmul_smoke("cuda", size=4)

    assert result.target == "nvidia_sm90"
    assert result.op_name == "tessera.matmul"
    assert result.runtime_status in {"ready", "artifact_only", "invalid_artifact"}
    assert result.telemetry["name"] == "compiler.gpu_smoke"


def test_apple_gpu_smoke_is_structured():
    result = run_matmul_smoke("apple_gpu", size=4)

    assert result.target == "apple_gpu"
    assert result.op_name == "tessera.matmul"
    assert result.runtime_status in {"ready", "artifact_only", "invalid_artifact"}
    assert result.telemetry["metadata"]["capability_version"].startswith("tessera.capabilities.")
