"""Checked four-member native quantizer/gate-up owner and portable package."""
from dataclasses import replace
import json
import pytest
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph,NativeQuantizedProductProgram,PreparedQuantizedProductProgram
from tests.unit.test_native_quantized_gate_up_program import source

pytestmark=[pytest.mark.compiler_route,pytest.mark.compiler_rocm]


@pytest.fixture
def package(production_compiler):
    return package_quantized_product_graph(source())


def test_four_member_owner_has_only_bf16_parent_and_result_buffers(package):
    owner=PreparedQuantizedProductProgram.__new__(PreparedQuantizedProductProgram)
    binding=owner._bind_abi(package)
    assert binding.argument_count==4 and binding.outputs==(9,)
    assert len(binding.steps)==4 and len(binding.storage)==10
    assert [list(s.inputs[:s.input_count]) for s in binding.steps]==[[0],[1,2,3],[4,6,5,7],[8]]
    assert [list(s.outputs[:s.output_count]) for s in binding.steps]==[[4,5],[6,7],[8],[9]]
    assert binding.storage[8].storage==binding.storage[9].storage=="bf16"
    assert binding.storage[8].shape==(3,34) and binding.storage[9].shape==(3,17)
    assert binding.storage[8].bytes==3*34*2
    assert list(binding.steps[3].scalars[:binding.steps[3].scalar_count])==[51]
    assert list(binding.steps[3].geometry)==[1,1,1,256,1,1]


def test_four_member_portable_restore_keeps_native_stage_order_without_compiler(package,monkeypatch):
    record=package.to_manifest()
    assert record["schema"]=="tessera.rocm.quantized_product_package.v3"
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    restored=NativeQuantizedProductProgram.from_manifest(record)
    assert restored.binding_digest==package.binding_digest
    assert len(restored.native_stages)==len(restored.schedule_irs)==4
    assert restored.activation_recipe.graph_ir==package.activation_recipe.graph_ir


def test_resealed_package_cannot_remove_activation(package):
    changed=replace(package,activation=None,activation_recipe=None)
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError,match="activation schema/member"):
        changed.validate()


@pytest.mark.parametrize("mutation",["parent_dtype","parent_bytes","early_death","wrong_output","wrong_step"])
def test_resealed_four_member_frame_refuses_storage_or_lifetime_corruption(package,mutation):
    plan=json.loads(package.program_json)
    if mutation=="parent_dtype":plan["buffers"][8]["storage"]="f32"
    elif mutation=="parent_bytes":plan["buffers"][8]["bytes"]*=2
    elif mutation=="early_death":plan["buffers"][8]["last_read"]=2
    elif mutation=="wrong_output":plan["output"]=8
    else:plan["steps"][3]["inputs"]=[7]
    changed=replace(package,program_json=json.dumps(plan))
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError):changed.validate()


def test_individually_valid_activation_cannot_overrun_the_shared_parent(package):
    from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    from tessera.compiler.rocm_parent_view_native import ParentViewRecipe,package_parent_view_recipe
    from tests.unit.test_native_nvfp4_parent_view_program import explicit_store_source
    graph=export_native_nvfp4_program(explicit_store_source(width=18)).project_member(2)
    tool=find_tessera_opt()
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    recipe=ParentViewRecipe(graph,schedule,tile)
    native=package_parent_view_recipe(recipe,pipeline_name="tessera-rocm-executable")
    plan=json.loads(package.program_json);plan["member_graphs"][3]=graph
    changed=replace(package,program_json=json.dumps(plan),activation_recipe=recipe,activation=native)
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError,match="activation/shared buffer"):
        changed.validate()


def test_public_four_member_capture_uses_native_semantic_admission(production_compiler):
    from benchmarks.rocm.benchmark_public_quantized_gate_up import make_gate_up
    from tessera.compiler.trace import trace,to_graph_ir_module
    from tessera.compiler.rocm_quantized_product_frontend import supports_quantized_product,package_traced_quantized_product
    result=trace(make_gate_up(),((3,48),"bf16"),((34,24),"uint8"),((34,3),"fp8_e4m3"),((2,),"fp64"),
        arg_names=["x","packed","weight_scales","globals_"],evaluate_catalog_outputs=False)
    module=to_graph_ir_module(result,name="gate_up",target="rocm_gfx1201")
    assert supports_quantized_product(module)
    program=package_traced_quantized_product(module)
    assert program.native.activation is not None
    assert program.manifest()["schema"]=="tessera.rocm.quantized_product_frontend.v3"
