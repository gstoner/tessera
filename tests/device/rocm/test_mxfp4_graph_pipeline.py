"""Exact-gfx1201 proof of a device-side FP8 producer/GEMM/BF16 consumer."""
from __future__ import annotations

import ctypes
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_graph_pipeline import (
    PackedFoldedGraphPipeline, PackedFoldedGraphPipelinePool,
)
from tessera.compiler.rocm_mxfp4_graph_tensor import ROCMGraphTensor
from tessera.compiler.rocm_mxfp4_packed_folded import (
    folded_oracle_from_packed,
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("m,n,k", [(65, 48, 64), (257, 80, 128)])
@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("producer_variant", ["block", "wave"])
def test_three_kernel_graph_produces_and_consumes_device_tensors(
    m: int, n: int, k: int, lossy: bool, producer_variant: str,
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    codes = np.resize(np.arange(16, dtype=np.uint8), (n, k))
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    if lossy:
        scales[0, ::3] = 116
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    x = np.resize(np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32), (m, k))
    x[0] *= 512.0
    folded = folded_oracle_from_packed(payload)

    def oracle(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_max = np.max(np.abs(value), axis=1)
        token_scale = np.maximum(1.0, row_max / np.float32(448.0))
        quantized = (value / token_scale[:, None]).astype(ml_dtypes.float8_e4m3fn)
        fp8 = quantized.view(np.uint8)
        acc = quantized.astype(np.float32) @ mx.folded_weights(folded).T
        bf16 = (acc * token_scale[:, None]).astype(ml_dtypes.bfloat16)
        relu = np.maximum(bf16.astype(np.float32), 0).astype(ml_dtypes.bfloat16)
        return fp8, token_scale, relu

    with PackedFoldedGraphPipeline(
        package, payload, m, producer_variant=producer_variant,
    ) as pipeline:
        pipeline.upload_fp32(x)
        pipeline.capture()
        assert pipeline.receipt()["capture_nodes"] == (0, 0, 0)
        pipeline.replay()
        actual = pipeline.read_final()
        expected_fp8, expected_scale, expected = oracle(x)
        np.testing.assert_array_equal(actual, expected)
        a = np.empty((m, k), dtype=np.uint8)
        a_scale = np.empty(m, dtype=np.float32)
        leases = pipeline.buffers()
        hip = pipeline._hip
        assert hip.hipMemcpy(
            a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(leases["a"].pointer),
            a.nbytes, 2,
        ) == 0
        assert hip.hipMemcpy(
            a_scale.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(leases["a_scale"].pointer), a_scale.nbytes, 2,
        ) == 0
        np.testing.assert_array_equal(a, expected_fp8)
        np.testing.assert_array_equal(a_scale, expected_scale)

        changed = np.ascontiguousarray(-x)
        assert hip.hipMemcpyAsync(
            ctypes.c_void_p(pipeline.input_pointer),
            changed.ctypes.data_as(ctypes.c_void_p), changed.nbytes, 1,
            ctypes.c_void_p(pipeline.stream_pointer),
        ) == 0
        pipeline.mark_device_input_ready(stream_pointer=pipeline.stream_pointer)
        pipeline.replay()
        np.testing.assert_array_equal(pipeline.read_final(), oracle(changed)[2])
        assert pipeline.receipt()["graph_replays"] == 2
        assert pipeline.receipt()["automatic_selection"] is False
    with pytest.raises(RuntimeError, match="closed"):
        _ = pipeline.input_pointer


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_exact_m_pool_keeps_shape_graphs_and_pointers_distinct() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    n, k = 48, 64
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(np.ones((n, k), dtype=np.uint8)),
        np.full((k // 32, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )

    def package_for_m(m: int, p: object) -> object:
        return package_mxfp4_packed_folded_prefill(
            m, p, permute_decode=True, batched_loads=True,
        )

    with PackedFoldedGraphPipelinePool(payload, package_for_m, max_shapes=2) as pool:
        first = pool.get(65)
        second = pool.get(129)
        assert pool.get(65) is first
        assert first.input_pointer != second.input_pointer
        assert first.final_output_pointer != second.final_output_pointer
        for pipeline, m in ((first, 65), (second, 129), (first, 65)):
            if not pipeline.receipt()["graph_captures"]:
                pipeline.upload_fp32(np.ones((m, k), dtype=np.float32))
                pipeline.capture()
            pipeline.replay()
            np.testing.assert_array_equal(
                pipeline.read_final(),
                np.full((m, n), k * 0.5, dtype=ml_dtypes.bfloat16),
            )
        assert pool.receipt()["active_m"] == [65, 129]
        with pytest.raises(RuntimeError, match="shape budget exhausted"):
            pool.get(257)
    with pytest.raises(RuntimeError, match="closed"):
        _ = first.input_pointer


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_model_owned_tensors_survive_graph_and_allocator_pressure() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(np.ones((n, k), dtype=np.uint8)),
        np.full((k // 32, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    x = np.ones((m, k), dtype=np.float32)
    with (ROCMGraphTensor((m, k), np.float32) as source,
          ROCMGraphTensor((m, n), ml_dtypes.bfloat16) as result):
        with PackedFoldedGraphPipeline(
            package, payload, m, input_tensor=source, output_tensor=result,
            producer_variant="wave",
        ) as pipeline:
            assert pipeline.receipt()["io_ownership"] == "model_borrowed"
            assert pipeline.input_pointer == source.pointer
            assert pipeline.final_output_pointer == result.pointer
            hip = pipeline._hip
            stream = ctypes.c_void_p(pipeline.stream_pointer)
            assert hip.hipMemcpyAsync(
                ctypes.c_void_p(source.pointer), x.ctypes.data_as(ctypes.c_void_p),
                x.nbytes, 1, stream,
            ) == 0
            pipeline.synchronize()
            pipeline.mark_device_input_ready(stream_pointer=pipeline.stream_pointer)
            pipeline.capture()
            with pytest.raises(RuntimeError, match="borrowed"):
                source.close()
            stable = (source.pointer, result.pointer)
            for _ in range(8):
                with ROCMGraphTensor((m, k), np.float32):
                    pipeline.replay()
                    np.testing.assert_array_equal(
                        pipeline.read_final(),
                        np.full((m, n), k * 0.5, dtype=ml_dtypes.bfloat16),
                    )
                assert (source.pointer, result.pointer) == stable
        assert source.borrowers == result.borrowers == 0
