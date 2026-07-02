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


def test_nvidia_features_match_cuda_matrix():
    """String-alias capability path is matrix-backed truth: no NVIDIA capability
    entry may advertise a feature the CUDA 13.3 matrix marks `not_supported`.

    Regression guard for consumer Blackwell sm_120 wrongly listing Hopper
    `wgmma` and datacenter `tcgen05`/`tcgen05_pair`/`tmem` in its `features`
    tuple — the single source of truth is `gpu_target._CUDA_13_3_FEATURES`
    (via `cuda_feature_status`).  Non-matrix marker tokens (e.g. `cuda_13_3`)
    are skipped.
    """
    from tessera.compiler.gpu_target import ISA, cuda_feature_status

    name_to_isa = {
        "sm80": ISA.SM_80,
        "sm90": ISA.SM_90,
        "sm100": ISA.SM_100,
        "sm120": ISA.SM_120,
    }
    offenders: list[str] = []
    for alias, isa in name_to_isa.items():
        cap = get_target_capability(alias)
        for feat in cap.features:
            try:
                status = cuda_feature_status(isa, feat)
            except KeyError:
                continue  # non-matrix marker (e.g. "cuda_13_3")
            if status != "ready":
                offenders.append(
                    f"{cap.name}: advertises {feat!r} but the CUDA matrix "
                    f"says {status!r}"
                )
    assert not offenders, (
        "NVIDIA capability `features` drift from the CUDA feature matrix "
        "(gpu_target._CUDA_13_3_FEATURES):\n  " + "\n  ".join(offenders)
    )

    # Positive lock: consumer sm_120 must NOT advertise the datacenter/Hopper
    # tensor-core flags (it is not a superset of sm_100).
    sm120_features = get_target_capability("sm120").features
    for banned in ("wgmma", "wgmma_sparse", "tcgen05", "tcgen05_pair", "tmem"):
        assert banned not in sm120_features, (
            f"consumer sm_120 must not advertise {banned!r}"
        )


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
