from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, tensor_ir_type
from tessera.compiler.nvidia_rematerialization import (
    apply_nvidia_rematerialization_policy,
    resolve_nvidia_rematerialization_policy,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    ROOT / "benchmarks/baselines/nvidia_sm120_training_memory_foundation.json"
)


def test_training_memory_baseline_is_resource_and_cache_complete() -> None:
    payload = json.loads(BASELINE.read_text())
    assert payload["schema"] == "tessera.nvidia.training-memory-foundation.v2"
    assert payload["device"] == "sm_120"
    assert payload["host_environment"] in {"wsl2", "native_linux"}
    assert payload["selector_changed"] is False
    assert payload["policy"]["discard_first_compile_launch"] is True
    operations = {row["operation"] for row in payload["rows"]}
    assert {
        "rmsnorm_backward",
        "layernorm_backward",
        "mse_backward",
        "mae_backward",
        "huber_backward",
        "smooth_l1_backward",
        "bce_backward",
        "cross_entropy_backward",
        "sgd",
        "momentum",
        "nesterov",
        "adam",
        "adamw",
        "fused_mse_sgd",
        "fused_mse_adamw",
        "dynamic_shared_path_max",
        "dynamic_shared_local_expression",
        "mse_backward_f16",
        "mse_backward_bf16",
        "rmsnorm_backward_f16",
        "layernorm_backward_bf16",
        "adamw_f16",
        "adamw_bf16",
        "kl_backward",
        "js_backward",
        "broadcast_gradient_reduce",
    } == operations
    memory = payload["device_memory_policy"]
    assert 0 < memory["effective_capacity_bytes"] <= memory["capacity_bytes"]
    assert memory["model_state_bytes"] >= memory["model_parameter_bytes"]
    assert memory["activation_budget_bytes"] >= 0
    for row in payload["rows"]:
        assert len(row["device_event_ns"]) == 2
        assert len(row["end_to_end_ns"]) == 2
        assert row["resource_fingerprint"]
        assert len(row["artifact_identity"]["image_digest"]) == 64
        assert len(row["artifact_identity"]["descriptor_digest"]) == 64
        assert row["artifact_identity"]["compiler_fingerprint"]
        assert row["artifact_identity"]["toolchain_fingerprint"]
        assert row["resources"]["registers_per_thread"] > 0
        assert row["resources"]["active_blocks_per_sm"] > 0
        assert row["resources"]["local_memory_bytes"] == 0
        assert row["compile_state"]["warm"] == "warm_cache"
        assert row["compile_state"]["same_image"] is True
        assert row["selector_changed"] is False


def test_fused_comparisons_retain_non_promoting_disposition() -> None:
    payload = json.loads(BASELINE.read_text())
    fused = {
        row["operation"]: row["comparison"]
        for row in payload["rows"]
        if row["operation"].startswith("fused_")
    }
    assert set(fused) == {"fused_mse_sgd", "fused_mse_adamw"}
    for comparison in fused.values():
        assert len(comparison["composed_device_event_ns"]) == 2
        assert len(comparison["composed_end_to_end_ns"]) == 2
        assert comparison["disposition"] == "evidence_only_no_production_selector"


def test_cuda_model_device_policy_stamps_shared_rematerialization_contract() -> None:
    module = GraphIRModule(
        functions=[
            GraphIRFunction(
                name="train",
                args=[
                    IRArg("weight", tensor_ir_type(("1024", "1024"), "fp16")),
                    IRArg("dynamic_weight", tensor_ir_type(("*",), "bf16")),
                    IRArg("x", tensor_ir_type(("1024",), "fp16")),
                ],
            )
        ]
    )
    policies = apply_nvidia_rematerialization_policy(
        module,
        parameter_names={"weight", "dynamic_weight"},
        parameter_bytes_bounds={"dynamic_weight": 1 << 20},
        capacity_bytes=32 << 20,
        free_bytes=24 << 20,
        reserve_basis_points=1000,
        gradient_copies=1,
        optimizer_state_copies=2,
        persistent_bytes=1 << 20,
    )
    policy = policies["train"]
    assert policy.device_capacity_bytes == 32 << 20
    assert policy.effective_capacity_bytes == 24 << 20
    assert policy.model_parameter_bytes == 3 << 20
    assert policy.model_state_bytes == 13 << 20
    assert policy.activation_budget_bytes == (24 << 20) * 9 // 10 - (13 << 20)
    mlir = module.to_mlir(verify=False)
    assert "tessera.model.parameter" in mlir
    assert "tessera.model.parameter_bytes_bound = 1048576 : i64" in mlir
    assert "tessera.device_memory_capacity_bytes = 25165824 : i64" in mlir
    assert "tessera.model_optimizer_state_copies = 2 : i32" in mlir


def test_cuda_model_device_policy_rejects_unsafe_capacity_contracts() -> None:
    with pytest.raises(ValueError, match="reserve_basis_points"):
        resolve_nvidia_rematerialization_policy(
            model_parameter_bytes=1, capacity_bytes=1024, reserve_basis_points=10001
        )
    module = GraphIRModule(
        functions=[
            GraphIRFunction(
                name="dynamic",
                args=[IRArg("weight", tensor_ir_type(("*",), "fp32"))],
            )
        ]
    )
    with pytest.raises(ValueError, match="requires parameter_bytes_bounds"):
        apply_nvidia_rematerialization_policy(
            module,
            parameter_names={"weight"},
            capacity_bytes=1024,
        )
