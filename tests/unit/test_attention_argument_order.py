"""The native checkpoint hash owns frontend argument-role mappings."""
from dataclasses import replace
import pytest
import numpy as np
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint

pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason="requires native compiler")


def traced(monkeypatch, order="k_bias_v_q"):
    import tessera.compiler.native_attention_program as native
    from benchmarks.nvidia.benchmark_attention_argument_order import function, ORDERS
    monkeypatch.setattr(native,"compile_attention_vjp_program",
                        lambda source,active,**kwargs:(source,active))
    values={name:np.ones(shape,np.float32) for name,shape in zip(
        ("q","k","v","bias"),((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,3,5)),strict=True)}
    return function(order,("bias","q"),True).compile_native_attention_vjp(
        *[values[name] for name in ORDERS[order]],compiler="unused")[0]


@pytest.mark.parametrize("backward",[False,True])
def test_native_export_binds_the_real_frontend_input_permutation(monkeypatch,backward):
    artifact=lower_generated_checkpoint(traced(monkeypatch),backward=backward)
    assert artifact.frontend_argument_indices==(3,0,2,1)
    assert "frontend_argument_indices = array<i64: 3, 0, 2, 1>" in artifact.tile_ir
    artifact.validate()
    with pytest.raises(ValueError,match="mapping"):
        replace(artifact,frontend_argument_indices=(0,1,2,3)).validate()


def test_schedule_replay_rejects_mutated_frontend_permutation(monkeypatch):
    artifact=lower_generated_checkpoint(traced(monkeypatch),backward=True)
    mutated=artifact.schedule_ir.replace("array<i64: 3, 0, 2, 1>","array<i64: 0, 1, 2, 3>")
    assert mutated!=artifact.schedule_ir
    with pytest.raises(RuntimeError,match="contract changed"):
        run_tessera_opt(find_tessera_opt(),mutated,"--tessera-schedule-to-tile")


@pytest.mark.parametrize("mapping",[(3,0,2,2),(4,0,2,1),(True,0,2,1),False])
def test_artifact_rejects_invalid_frontend_mapping(monkeypatch,mapping):
    artifact=lower_generated_checkpoint(traced(monkeypatch),backward=True)
    with pytest.raises(ValueError,match="mapping"):
        replace(artifact,frontend_argument_indices=mapping).validate()


@pytest.mark.parametrize("order,wrt,active",[
    ("k_bias_v_q",("bias","q"),[3,0]),
    ("bias_v_q_k",("v","bias","k","q"),[2,3,1,0]),
    ("v_q_k",("v","q"),[2,0]),
])
def test_permuted_frontend_capture_and_cotangents_execute_on_sm120(order,wrt,active):
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():pytest.skip("requires owning SM120 host")
    from benchmarks.nvidia.benchmark_attention_argument_order import run_case
    result=run_case((1,2,1,3,5,4,3),order,wrt,True,samples=3)
    assert result["native_gradient_result_indices"]==active
    assert result["max_abs_gradient_error"]<4e-5


def test_pruned_activity_uses_physical_roles_after_frontend_permutation(monkeypatch):
    artifact = lower_generated_checkpoint(traced(monkeypatch), backward=True, prune_inactive=True)
    assert artifact.frontend_argument_indices == (3,0,2,1)
    assert artifact.gradient_activity == (1,0,0,1)
    artifact.validate()
