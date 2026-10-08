"""Native physical storage selection must replay before checked packaging."""
from dataclasses import replace
import pytest
from tessera.compiler import scheduled_matmul as schedule, nvidia_native as native
from tests.unit.test_scheduled_matmul_consumers import _module

pytestmark = pytest.mark.skipif(schedule.find_tessera_opt() is None, reason="requires native compiler")


def graph(dtype="fp16", shape=(17,35,19), order="row_major", target="nvidia_sm120"):
    module=_module(target=target,dtype=dtype,shape=shape)
    module.functions[0].body[0].kwargs["rhs_storage_order"]=order
    return module


@pytest.mark.parametrize("dtype", ["fp16","bf16"])
@pytest.mark.parametrize("shape", [(16,16,8),(17,35,19),(256,1024,128)])
def test_row_storage_is_native_hash_and_typed_view_contract(dtype,shape):
    row=schedule.lower_scheduled_matmul(graph(dtype,shape),target="nvidia_sm120")
    col=schedule.lower_scheduled_matmul(_module(target="nvidia_sm120",dtype=dtype,shape=shape),target="nvidia_sm120")
    assert row.b_layout == "row_major"
    assert row.schedule_digest != col.schedule_digest
    assert 'b_layout = "row_major"' in row.schedule_ir
    assert 'role = "b", transpose' in row.tile_ir
    assert "tessera_nvidia.macro_cta_matmul" not in row.tile_ir
    schedule.verify_matmul_projection(row)
    with pytest.raises(ValueError,match="b_layout"):
        schedule.verify_matmul_projection(replace(row,b_layout="col_major"))


def test_replay_rejects_storage_change_after_hashing():
    row=schedule.lower_scheduled_matmul(graph(),target="nvidia_sm120")
    changed=row.schedule_ir.replace('b_layout = "row_major"', 'b_layout = "col_major"')
    with pytest.raises(RuntimeError,match="altered after hashing"):
        schedule.run_tessera_opt(schedule.find_tessera_opt(),changed,"--tessera-schedule-to-tile")


@pytest.mark.parametrize("order", ["diagonal",True])
def test_invalid_storage_order_is_not_silently_normalized(order):
    with pytest.raises(RuntimeError,match="rhs_storage_order"):
        schedule.lower_scheduled_matmul(graph(order=order),target="nvidia_sm120")


def test_sibling_target_refuses_unimplemented_storage_request():
    with pytest.raises(RuntimeError,match="rhs_storage_order"):
        schedule.lower_scheduled_matmul(graph(target="rocm_gfx1201"),target="rocm_gfx1201")


@pytest.mark.parametrize("dtype", ["fp16","bf16"])
def test_checked_descriptor_preserves_native_storage_selection(monkeypatch,dtype):
    row=schedule.lower_scheduled_matmul(graph(dtype),target="nvidia_sm120")
    monkeypatch.setattr(native,"_compile_tile_ir",
                        lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    package=native.package_scheduled_matmul(row,pipeline_name="tessera-nvidia-pipeline-sm120")
    expected=native.SM120_ROW_B_F16_ABI if dtype=="fp16" else native.SM120_ROW_B_BF16_ABI
    assert package.descriptor.abi_id == expected
    assert package.descriptor.buffers[1].layout == "row_major"
    assert package.descriptor.provenance["b_layout"] == "row_major"
