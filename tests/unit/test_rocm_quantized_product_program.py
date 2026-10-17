"""Production composed-package metadata and checked multi-output bindings."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from tessera.compiler.rocm_quantized_product_program import (
    _frame_shape,package_quantized_product_graph,PreparedQuantizedProductProgram,
)
from tests.unit.test_native_quantized_product_program import source

BASE = Path(__file__).resolve().parents[2]/"benchmarks/baselines/rocm_resident_quantized_product_20261017"


def reference_plan():
    return json.loads((BASE/"resident-product-3-17-32-final.json").read_text())["plan"]


def test_native_plan_projects_original_roles_and_static_capacity():
    roles,shape = _frame_shape(reference_plan())
    assert roles == [0,1,2]
    assert (shape.m,shape.n,shape.k,shape.output) == (3,17,32,"f32")


@pytest.mark.parametrize("mutation",["bool_id","bytes","private_owner","read_before_write","output_role","wiring","boolean_extent","extra_field"])
def test_plan_rejects_resealed_storage_and_lifetime_corruption(mutation):
    plan=deepcopy(reference_plan())
    if mutation=="bool_id":plan["buffers"][0]["id"]=False
    elif mutation=="bytes":plan["buffers"][3]["bytes"]+=1
    elif mutation=="private_owner":plan["buffers"][4]["ownership"]="readonly_input"
    elif mutation=="read_before_write":plan["buffers"][3]["first_write"]=1
    elif mutation=="output_role":plan["steps"][1]["outputs"]=[4]
    elif mutation=="wiring":plan["steps"][1]["inputs"][2]=2
    elif mutation=="boolean_extent":plan["buffers"][0]["shape"][0]=True
    else:plan["unexpected"]=True
    with pytest.raises(ValueError):_frame_shape(plan)


@pytest.fixture
def package(production_compiler):
    return package_quantized_product_graph(source())


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_production_package_binds_both_private_edges_without_a_compiler(package,monkeypatch):
    package.validate()
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    package.validate()
    owner=PreparedQuantizedProductProgram.__new__(PreparedQuantizedProductProgram)
    binding=owner._bind_abi(package)
    assert binding.argument_count==3 and binding.outputs==(5,)
    assert len(binding.steps)==2 and len(binding.storage)==6
    assert list(binding.steps[0].outputs[:2])==[3,4]
    assert list(binding.steps[1].inputs[:4])==[3,1,4,2]
    assert list(binding.steps[0].scalars[:1])==[6]
    assert list(binding.steps[1].scalars[:3])==[3,17,32]
    assert [b.ownership for b in binding.buffers]==[0,0,0,1,1,2]


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_package_validates_resealed_native_plan(package):
    plan=json.loads(package.program_json);plan["buffers"][4]["last_read"]=0
    changed=replace(package,program_json=json.dumps(plan))
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError,match="lifetime"):
        changed.validate()


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_portable_package_restores_with_compiler_disabled(package,monkeypatch):
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    record=package.to_manifest()
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    restored=NativeQuantizedProductProgram.from_manifest(record)
    restored.validate()
    assert restored.binding_digest==package.binding_digest
    assert [p.image.image_digest for p in restored.native_stages]==[p.image.image_digest for p in package.native_stages]


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("kind",["max","sum"])
def test_traced_candidate_uses_native_semantic_admission(kind):
    from tests.device.rocm.test_native_quantized_product_jit import make_product
    from tessera.compiler.trace import trace,to_graph_ir_module
    from tessera.compiler.rocm_quantized_product_frontend import supports_quantized_product
    result=trace(make_product(),((3,32),"bf16"),((17,32),"fp8_e4m3"),((2,17),"fp32"),
                 arg_names=["x","b","sb"],evaluate_catalog_outputs=False)
    module=to_graph_ir_module(result,name="product",target="rocm_gfx1201")
    module.functions[0].body[3].kwargs["kind"]=kind
    assert supports_quantized_product(module)==(kind=="max")


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
@pytest.mark.parametrize("stage",["graph_ir","schedule_ir","tile_ir","target_ir"])
def test_portable_parent_stage_mutation_refuses_before_provider(package,stage,monkeypatch):
    from tessera.compiler.rocm_quantized_product_frontend import TracedQuantizedProduct,runtime_artifact,execute_artifact
    from tessera import runtime as rt
    artifact=runtime_artifact(TracedQuantizedProduct(package,("x","b","sb")))
    artifact=replace(artifact,**{stage:getattr(artifact,stage)+"\n// changed parent witness"})
    def forbidden():raise AssertionError("provider must not be reached")
    monkeypatch.setattr(rt,"_load_rocm_native_movement_runtime",forbidden)
    with pytest.raises(ValueError,match="Graph|witness"):
        execute_artifact(artifact,[])


@pytest.mark.compiler_route
@pytest.mark.compiler_rocm
def test_individually_valid_wrong_shape_quantizer_cannot_bind_shared_buffers(package):
    from tessera.compiler.rocm_block_quant_native import lower_block_quant_text,package_block_quant_recipe
    from tests.unit.test_native_k16_quant_graph import source as quant_source
    # All member descriptors remain valid; only the composed edge is wrong.
    recipe=lower_block_quant_text(quant_source(3,64))
    other=package_block_quant_recipe(recipe)
    plan=json.loads(package.program_json)
    plan["member_graphs"][0]=recipe.graph_ir
    changed=replace(package,program_json=json.dumps(plan),quantizer_recipe=recipe,quantizer=other)
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError,match="producer/shared buffer"):
        changed.validate()
