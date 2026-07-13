from __future__ import annotations

from dataclasses import replace

from tessera.compiler import autodiff_promotion as promotion
from tessera.compiler import execution_matrix as em


def test_mock_collective_never_promotes_to_device_training():
    result = promotion.collective_promotion("mock_collective")
    assert not result.eligible
    assert result.status == "reference_only"


def test_rccl_without_exact_multi_device_fixture_remains_hardware_gated():
    result = promotion.collective_promotion("rccl")
    assert not result.eligible
    assert result.status == "hardware_gated"


def test_forward_row_cannot_prove_backward_training():
    row = em.lookup("rocm", "rocm_flash_attn_compiled")
    assert row is not None
    result = promotion.accelerator_backward_promotion(row)
    assert not result.eligible and result.status == "forward_only"


def test_exact_rocm_backward_satisfies_phase6_gate():
    row = em.lookup("rocm", "rocm_flash_attn_bwd_compiled")
    assert row is not None
    result = promotion.accelerator_backward_promotion(row)
    assert result.eligible
    assert result.status == "device_verified_jit"


def test_apple_gpu_requires_fresh_process_and_metal_provenance():
    base = em.lookup("rocm", "rocm_flash_attn_bwd_compiled")
    assert base is not None
    apple = replace(
        base,
        target="apple_gpu",
        evidence_target="apple_gpu_m4",
        execution_mode="metal_runtime",
    )
    assert not promotion.accelerator_backward_promotion(apple).eligible
    assert promotion.accelerator_backward_promotion(apple, fresh_process=True).eligible

