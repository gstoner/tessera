from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.nvidia_native import package_moe_kernels, package_replay_ssm_kernels
from tessera.runtime import _submit_nvidia_sm120_native
from tests._support.nvidia import nvidia_cuda_host_ready


@pytest.mark.hardware_nvidia
def test_replay_images_are_compiler_owned_and_session_persistent() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    decode, flush = package_replay_ssm_kernels(
        batch=2, channels=3, state_dim=4, capacity=8, async_slots=2,
        pipeline_name="tessera-nvidia-pipeline-sm120",
    )
    assert decode.descriptor.provenance["route"] == "output_only"
    assert flush.descriptor.provenance["route"] == "state_and_output"
    assert decode.descriptor.workspace.lifetime == "session"
    assert decode.descriptor.workspace.initialization == "preserve"
    assert decode.image.resource_record is not None
    assert flush.image.resource_record is not None
    assert "tile.replay_ssm_decode_kernel" in decode.tile_ir
    assert "tile.replay_ssm_decode_kernel" not in decode.target_ir


@pytest.mark.hardware_nvidia
def test_moe_images_execute_dispatch_combine_and_ragged_grouped_gemm() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    tokens, slots, hidden, experts, k, n = 4, 6, 5, 3, 4, 3
    packages = package_moe_kernels(
        num_tokens=tokens, num_slots=slots, hidden=hidden,
        expert_count=experts, expert_k=k, expert_n=n,
        group_offsets=(0, 1, 4, 6), pipeline_name="tessera-nvidia-pipeline-sm120",
    )
    dispatch, combine, grouped = packages
    token_of_slot = np.array([2, 0, 3, 1, 0, 2], np.int32)
    x = np.arange(tokens * hidden, dtype=np.float32).reshape(tokens, hidden) / 7.0
    dispatched = np.zeros((slots, hidden), np.float32)
    got = _submit_nvidia_sm120_native(
        dispatch.image, dispatch.descriptor,
        {"X": x, "token_of_slot": token_of_slot, "dispatched": dispatched},
        {"Tokens": tokens, "Slots": slots, "Hidden": hidden}, None,
    )
    np.testing.assert_array_equal(got, x[token_of_slot])

    weights = np.array([0.2, 0.5, 0.7, 0.3, 0.5, 0.8], np.float32)
    combined = np.zeros((tokens, hidden), np.float32)
    got = _submit_nvidia_sm120_native(
        combine.image, combine.descriptor,
        {"partials": dispatched, "token_of_slot": token_of_slot,
         "combine_weights": weights, "O": combined},
        {"Tokens": tokens, "Slots": slots, "Hidden": hidden}, None,
    )
    expected = np.zeros_like(combined)
    for slot, token in enumerate(token_of_slot):
        expected[token] += dispatched[slot] * weights[slot]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

    grouped_x = np.arange(slots * k, dtype=np.float32).reshape(slots, k) / 11.0
    expert_w = np.arange(experts * k * n, dtype=np.float32).reshape(experts, k, n) / 13.0
    offsets = np.array([0, 1, 4, 6], np.int32)
    grouped_o = np.zeros((slots, n), np.float32)
    got = _submit_nvidia_sm120_native(
        grouped.image, grouped.descriptor,
        {"X": grouped_x, "W": expert_w, "group_offsets": offsets, "O": grouped_o},
        {"GroupedTokens": slots, "K": k, "N": n, "Experts": experts}, None,
    )
    expected = np.empty_like(grouped_o)
    for expert in range(experts):
        expected[offsets[expert]:offsets[expert + 1]] = (
            grouped_x[offsets[expert]:offsets[expert + 1]] @ expert_w[expert]
        )
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-5)
    for package in packages:
        assert package.image.resource_record is not None
        assert package.image.compile_state in {"cold", "warm_cache"}


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["fp16", "bf16"])
def test_moe_images_cover_low_precision_storage_with_f32_accumulation(storage) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    dtype = np.float16 if storage == "fp16" else ml_dtypes.bfloat16
    tokens, slots, hidden, experts, k, n = 3, 5, 4, 2, 4, 3
    dispatch, combine, grouped = package_moe_kernels(
        num_tokens=tokens, num_slots=slots, hidden=hidden,
        expert_count=experts, expert_k=k, expert_n=n,
        group_offsets=(0, 2, 5), pipeline_name="tessera-nvidia-pipeline-sm120",
        storage=storage,
    )
    token = np.array([2, 0, 1, 0, 2], np.int32)
    x = (np.arange(tokens * hidden).reshape(tokens, hidden) / 9.0).astype(dtype)
    dispatched = np.zeros((slots, hidden), dtype=dtype)
    got = _submit_nvidia_sm120_native(
        dispatch.image, dispatch.descriptor,
        {"X": x, "token_of_slot": token, "dispatched": dispatched},
        {"Tokens": tokens, "Slots": slots, "Hidden": hidden}, None,
    )
    np.testing.assert_array_equal(got, x[token])
    weights = np.array([0.2, 0.3, 0.4, 0.5, 0.6], np.float32)
    combined = np.zeros((tokens, hidden), dtype=dtype)
    got = _submit_nvidia_sm120_native(
        combine.image, combine.descriptor,
        {"partials": dispatched, "token_of_slot": token,
         "combine_weights": weights, "O": combined},
        {"Tokens": tokens, "Slots": slots, "Hidden": hidden}, None,
    )
    expected = np.zeros((tokens, hidden), np.float32)
    for slot, tok in enumerate(token):
        expected[tok] += dispatched[slot].astype(np.float32) * weights[slot]
    np.testing.assert_allclose(got.astype(np.float32), expected, rtol=5e-3, atol=5e-3)
    gx = (np.arange(slots * k).reshape(slots, k) / 13.0).astype(dtype)
    w = (np.arange(experts * k * n).reshape(experts, k, n) / 17.0).astype(dtype)
    offsets = np.array([0, 2, 5], np.int32); out = np.zeros((slots, n), dtype=dtype)
    got = _submit_nvidia_sm120_native(
        grouped.image, grouped.descriptor,
        {"X": gx, "W": w, "group_offsets": offsets, "O": out},
        {"GroupedTokens": slots, "K": k, "N": n, "Experts": experts}, None,
    )
    expected = np.empty((slots, n), np.float32)
    for expert in range(experts):
        expected[offsets[expert]:offsets[expert + 1]] = (
            gx[offsets[expert]:offsets[expert + 1]].astype(np.float32)
            @ w[expert].astype(np.float32)
        )
    np.testing.assert_allclose(got.astype(np.float32), expected, rtol=8e-3, atol=8e-3)
    for package in (dispatch, combine, grouped):
        assert package.descriptor.provenance["storage"] in {"f16", "bf16"}
        assert package.descriptor.provenance["accum"] == "f32"
