from __future__ import annotations

import pytest

from tessera.compiler.apple_fragment import (
    AppleFragmentError,
    AppleTilePromotionEvidence,
    select_apple_simdgroup_fragment,
    select_apple_tile_promotion,
)
from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import (
    dispatch_apple_simdgroup_tile_f16,
    materialize_apple_simdgroup_tile_msl,
    validate_steel_gemm_structure,
)


@pytest.mark.parametrize("dtype", ["fp16", "f16", "bf16"])
def test_apple7_simdgroup_fragment_is_exact_physical_contract(dtype):
    fragment = select_apple_simdgroup_fragment(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), dtype)
    assert (fragment.m, fragment.n, fragment.k) == (8, 8, 8)
    assert fragment.lanes == 32
    assert fragment.threadgroup == (32, 1, 1)
    assert fragment.accumulator_dtype == "fp32"
    assert fragment.as_metadata_dict()["family"] == "simdgroup_matrix"


def test_fragment_rejects_non_matrix_storage_and_accumulator():
    target = AppleGPUTargetProfile(AppleGPUArch.APPLE7)
    with pytest.raises(AppleFragmentError, match="UNSUPPORTED_DTYPE"):
        select_apple_simdgroup_fragment(target, "fp32")
    with pytest.raises(AppleFragmentError, match="UNSUPPORTED_ACCUMULATOR"):
        select_apple_simdgroup_fragment(target, "fp16", accumulator_dtype="fp16")


def _promotion_evidence(route, medians, *, counter_supported=False, **overrides):
    values = dict(
        route=route,
        dtype="fp16",
        shape=(32, 16, 32),
        timing_domain="kernel",
        native_gpu=True,
        numerically_validated=True,
        placement_validated=True,
        run_medians_ns=medians,
        resource_record={"total_threadgroup_bytes": 4352},
        counter_sampling_supported=counter_supported,
        counter_timestamp_deltas=(None, None),
    )
    values.update(overrides)
    return AppleTilePromotionEvidence(**values)


def test_tile_promotion_requires_proof_and_two_stable_domain_wins():
    mps = _promotion_evidence("mps", (100, 100))
    simd = _promotion_evidence("simdgroup_matrix", (94, 94))
    assert select_apple_tile_promotion(mps, simd) == "simdgroup_matrix"
    assert select_apple_tile_promotion(
        mps, _promotion_evidence("simdgroup_matrix", (94, 97))) == "mps"
    assert select_apple_tile_promotion(
        mps, _promotion_evidence("simdgroup_matrix", (80, 80), placement_validated=False)) == "mps"
    assert select_apple_tile_promotion(
        mps, _promotion_evidence("simdgroup_matrix", (80, 80),
                                 counter_sampling_supported=None)) == "mps"


def test_tile_promotion_requires_real_counter_deltas_when_capable():
    mps = _promotion_evidence("mps", (100, 100), counter_supported=True,
                              counter_timestamp_deltas=(20, 20))
    simd = _promotion_evidence("simdgroup_matrix", (80, 80), counter_supported=True,
                               counter_timestamp_deltas=(10, 11))
    assert select_apple_tile_promotion(mps, simd) == "simdgroup_matrix"
    assert select_apple_tile_promotion(
        mps, _promotion_evidence("simdgroup_matrix", (80, 80), counter_supported=True,
                                 counter_timestamp_deltas=(None, None))) == "mps"


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_target_selected_fragment_materializes_steel_msl_with_ragged_store(dtype):
    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), dtype, 32, 32, 16)

    assert artifact.fragment.storage_dtype == {"f16": "fp16", "bf16": "bf16"}[dtype]
    assert artifact.fragment.threadgroup == (32, 1, 1)
    assert artifact.shape.m == 32
    assert artifact.resources.threadgroup == artifact.fragment.threadgroup
    assert artifact.resources.simdgroup_lanes == artifact.fragment.lanes
    assert artifact.resources.total_threadgroup_bytes == 4352
    assert artifact.resources.total_threadgroup_bytes <= (
        artifact.resources.target_threadgroup_capacity_bytes)
    assert validate_steel_gemm_structure(
        artifact.msl, dtype=artifact.fragment.storage_dtype,
        partial_edge=True, double_buffer=True).ok
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in artifact.msl
    assert "copy only valid elements" in artifact.msl


def test_target_selected_materializer_rejects_non_fragment_tile_extent():
    with pytest.raises(ValueError, match="positive multiples"):
        materialize_apple_simdgroup_tile_msl(
            AppleGPUTargetProfile(AppleGPUArch.APPLE7), "f16", 30, 32, 16)


def test_target_selected_materializer_rejects_threadgroup_memory_overflow():
    with pytest.raises(AppleFragmentError, match="THREADGROUP_MEMORY_EXCEEDED"):
        materialize_apple_simdgroup_tile_msl(
            AppleGPUTargetProfile(AppleGPUArch.APPLE7, threadgroup_memory_bytes=4096),
            "f16", 32, 32, 16)


def test_tile_runtime_abi_receives_selected_source_and_reports_no_fallback(monkeypatch):
    import numpy as np
    from tessera import _apple_gpu_dispatch

    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), "f16", 32, 32, 16)
    calls = []

    def unavailable(*args):
        calls.append(args)
        return 0

    monkeypatch.setattr(
        _apple_gpu_dispatch, "bind_registered", lambda symbol: unavailable)
    out, native = dispatch_apple_simdgroup_tile_f16(
        artifact, np.ones((8, 8), np.float16), np.ones((8, 8), np.float16))

    assert out is None and native is False
    assert calls[0][0] == artifact.msl.encode()
    assert calls[0][1] == artifact.entry.encode()
    assert calls[0][-1].value == 32


def test_tile_runtime_provenance_retains_source_and_resource_record(monkeypatch):
    import numpy as np
    from tessera import _apple_gpu_dispatch

    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), "f16", 32, 32, 16)
    monkeypatch.setattr(_apple_gpu_dispatch, "bind_registered", lambda _symbol: lambda *_args: 0)
    out, native, record = dispatch_apple_simdgroup_tile_f16(
        artifact, np.ones((8, 8), np.float16), np.ones((8, 8), np.float16),
        return_provenance=True)
    assert out is None and native is False
    assert record.execution_mode == "reference_cpu"
    assert len(record.source_sha256) == 64
    assert record.resources["total_threadgroup_bytes"] == 4352


def _to_bf16(a):
    import numpy as np
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(a):
    import numpy as np
    return (np.asarray(a, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("dtype,shape", [
    ("f16", (8, 8, 8)),
    ("bf16", (8, 8, 8)),
    ("f16", (13, 11, 16)),
])
def test_apple7_simdgroup_tile_executes_and_missing_binding_is_explicit(
    monkeypatch, dtype, shape,
):
    import numpy as np
    from tessera import _apple_gpu_dispatch
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), dtype, 32, 32, 16)
    m, n, k = shape
    af = np.arange(m * k, dtype=np.float32).reshape(m, k) / 16
    bf = np.arange(k * n, dtype=np.float32).reshape(k, n) / 32
    a, b = (_to_bf16(af), _to_bf16(bf)) if dtype == "bf16" else (
        af.astype(np.float16), bf.astype(np.float16))
    out, native, record = dispatch_apple_simdgroup_tile_f16(
        artifact, a, b, return_provenance=True)
    assert native is True
    assert record.device_time_ns is not None and record.device_time_ns > 0
    if record.counter_sampling_supported:
        assert record.counter_timestamp_delta is not None
        assert record.counter_timestamp_delta > 0
    else:
        assert record.counter_timestamp_delta is None
    expected = _from_bf16(a) @ _from_bf16(b) if dtype == "bf16" else (
        a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(out, expected,
                               rtol=1e-5, atol=1e-5)

    monkeypatch.setattr(_apple_gpu_dispatch, "bind_registered", lambda _symbol: None)
    fallback, fallback_native = dispatch_apple_simdgroup_tile_f16(artifact, a, b)
    assert fallback is None
    assert fallback_native is False
