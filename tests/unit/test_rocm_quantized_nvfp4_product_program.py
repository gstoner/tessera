"""Native three-member package, portable binding and input validation."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import ml_dtypes
import pytest
from tessera.compiler.rocm_quantized_product_program import (
    _frame_shape, package_quantized_product_graph, NativeQuantizedProductProgram,
    PreparedQuantizedProductProgram,
)
from tests.unit.test_native_quantized_nvfp4_product_program import composed_source

pytestmark = [pytest.mark.compiler_route,pytest.mark.compiler_rocm]


@pytest.fixture
def package(production_compiler):
    return package_quantized_product_graph(composed_source())


def test_three_member_native_owner_keeps_all_four_intermediate_planes_private(package):
    package.validate()
    owner=PreparedQuantizedProductProgram.__new__(PreparedQuantizedProductProgram)
    binding=owner._bind_abi(package)
    assert binding.argument_count==4 and binding.outputs==(8,)
    assert len(binding.steps)==3 and len(binding.storage)==9
    assert [list(step.outputs[:step.output_count]) for step in binding.steps]==[[4,5],[6,7],[8]]
    assert [list(step.inputs[:step.input_count]) for step in binding.steps]==[[0],[1,2,3],[4,6,5,7]]
    assert binding.steps[1].scalar_count==0
    assert list(binding.steps[1].geometry)==[1,1,1,256,1,1]
    assert [b.ownership for b in binding.buffers]==[0]*4+[1]*4+[2]
    assert binding.storage[1].numpy_dtype==np.dtype("uint8")
    assert binding.storage[3].numpy_dtype==np.dtype("float64")


def test_three_member_portable_roundtrip_needs_no_compiler(package,monkeypatch):
    record=package.to_manifest()
    assert record["schema"]=="tessera.rocm.quantized_product_package.v2"
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    restored=NativeQuantizedProductProgram.from_manifest(record)
    assert restored.binding_digest==package.binding_digest
    assert len(restored.native_stages)==3


@pytest.mark.parametrize("mutation",["missing_expansion","global_shape","read_before_write","lost_plane","old_schema"])
def test_three_member_frame_rejects_resealed_mutations(package,mutation):
    plan=deepcopy(json.loads(package.program_json))
    if mutation=="missing_expansion":plan["steps"].pop(1)
    elif mutation=="global_shape":plan["buffers"][3]["shape"]=[2]
    elif mutation=="read_before_write":plan["buffers"][6]["first_write"]=2
    elif mutation=="lost_plane":plan["steps"][2]["inputs"][3]=5
    else:plan["schema"]="tessera.native.quantized_product.v1"
    with pytest.raises(ValueError):_frame_shape(plan)


def test_expansion_member_cannot_be_removed_by_resealing(package):
    changed=replace(package,expansion=None)
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError,match="schema/member"):
        changed.validate()


@pytest.mark.parametrize("kind",["nan_scale","negative_scale","zero_global","overflow","underflow"])
def test_invalid_expansion_roots_reject_before_native_marshalling(package,kind,monkeypatch):
    from tessera.compiler.native_scaled_program import PreparedScaledProgram
    values=[np.zeros((3,32),dtype=ml_dtypes.bfloat16),np.zeros((17,16),np.uint8),
            np.ones((17,2),dtype=ml_dtypes.float8_e4m3fn),np.ones((1,),np.float64)]
    if kind=="nan_scale":values[2][0,0]=np.nan
    elif kind=="negative_scale":values[2][0,0]=-1
    elif kind=="zero_global":values[3][0]=0
    elif kind=="overflow":values[3][0]=np.finfo(np.float64).max
    else:values[3][0]=np.nextafter(0.,1.)
    owner=PreparedQuantizedProductProgram.__new__(PreparedQuantizedProductProgram)
    owner.package=package
    def forbidden(*a,**kw):raise AssertionError("invalid expansion roots reached native marshalling")
    monkeypatch.setattr(PreparedScaledProgram,"_inputs",forbidden)
    with pytest.raises(ValueError,match="NVFP4"):
        owner._inputs(values)


def test_public_capture_selects_only_native_four_root_contract(production_compiler):
    from benchmarks.rocm.benchmark_public_quantized_nvfp4_product import make_product
    from tessera.compiler.trace import trace,to_graph_ir_module
    from tessera.compiler.rocm_quantized_product_frontend import supports_quantized_product,package_traced_quantized_product
    result=trace(make_product(),((3,32),"bf16"),((17,16),"uint8"),((17,2),"fp8_e4m3"),((1,),"fp64"),
                 arg_names=["x","packed","weight_scales","globals_"],evaluate_catalog_outputs=False)
    module=to_graph_ir_module(result,name="product",target="rocm_gfx1201")
    assert supports_quantized_product(module)
    program=package_traced_quantized_product(module)
    assert program.native.expansion is not None
    assert program.manifest()["schema"]=="tessera.rocm.quantized_product_frontend.v2"
