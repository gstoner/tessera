from __future__ import annotations

import pytest

from tessera.compiler.capabilities import (
    CAPABILITY_REGISTRY_VERSION,
    get_target_capability,
    normalize_target,
    runtime_status,
    supports_op,
)
from tessera.compiler.matmul_pipeline import normalize_target_kind


def test_capability_registry_normalizes_existing_target_aliases():
    assert normalize_target("cuda") == "nvidia_sm90"
    assert normalize_target("x86_64") == "cpu"
    assert normalize_target("m-series-gpu") == "apple_gpu"
    assert normalize_target_kind("sm100") == "nvidia_sm100"

    with pytest.raises(ValueError):
        normalize_target("quantum_waffle")


def test_capability_registry_reports_runtime_status_by_op():
    assert runtime_status("cpu", "tessera.matmul") == "ready"
    assert runtime_status("apple_gpu", "tessera.gelu") == "ready"
    assert runtime_status("rocm", "tessera.flash_attn") == "artifact_only"

    result = supports_op("apple_gpu", "tessera.gelu", dtype="fp32", rank=2)
    assert result.supported
    assert result.capability_version == CAPABILITY_REGISTRY_VERSION
    assert "Apple GPU" in result.reason


def test_target_capability_shape_is_shared_metadata():
    cap = get_target_capability("hopper")

    assert cap.name == "nvidia_sm90"
    assert cap.runtime_backend == "cuda"
    assert cap.default_runtime_status == "artifact_only"
    assert "wgmma" in cap.features
