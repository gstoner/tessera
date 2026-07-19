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
