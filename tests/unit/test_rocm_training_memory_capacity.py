from __future__ import annotations

from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    tensor_ir_type,
)
from tessera.compiler.rocm_rematerialization import (
    apply_rocm_rematerialization_policy,
)


def test_rocm_model_capacity_stamps_shared_rematerialization_contract() -> None:
    module = GraphIRModule(functions=[
        GraphIRFunction(
            name="train",
            args=[
                IRArg("weight", tensor_ir_type(("512", "1024"), "bf16")),
                IRArg("dynamic", tensor_ir_type(("*",), "fp32")),
                IRArg("activation", tensor_ir_type(("512",), "bf16")),
            ],
        )
    ])
    policies = apply_rocm_rematerialization_policy(
        module,
        parameter_names={"weight", "dynamic"},
        parameter_bytes_bounds={"dynamic": 2 << 20},
        capacity_bytes=16 << 20,
        free_bytes=12 << 20,
        reserve_basis_points=1250,
        gradient_copies=1,
        optimizer_state_copies=1,
        persistent_bytes=1 << 20,
    )
    policy = policies["train"]
    assert policy.effective_capacity_bytes == 12 << 20
    assert policy.model_parameter_bytes == 3 << 20
    assert policy.model_state_bytes == 10 << 20
    assert policy.activation_budget_bytes == (12 << 20) * 8750 // 10000 - (10 << 20)
    text = module.to_mlir(verify=False)
    assert "tessera.device_memory_capacity_bytes = 12582912 : i64" in text
    assert "tessera.model.parameter_bytes_bound = 2097152 : i64" in text


def test_rocm_capacity_query_uses_retained_hip_context() -> None:
    from tessera import runtime

    if not runtime._rocm_wmma_runtime_available():
        return
    envelope = runtime._rocm_device_memory_envelope()
    assert 0 < envelope["free_bytes"] <= envelope["capacity_bytes"]
