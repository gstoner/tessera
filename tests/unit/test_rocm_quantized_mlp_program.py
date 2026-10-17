"""Single native-owner binding for both quantizers and NVFP4 products."""
from dataclasses import replace
import json
import pytest
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram,PreparedQuantizedMLPProgram
from tests.unit.test_native_quantized_mlp_program import source

pytestmark=[pytest.mark.compiler_route,pytest.mark.compiler_rocm]


@pytest.fixture
def package(production_compiler):
    return package_quantized_product_graph(source())


def test_full_mlp_one_owner_has_all_seven_steps_and_resident_activation(package):
    assert isinstance(package,NativeQuantizedMLPProgram)
    owner=PreparedQuantizedMLPProgram.__new__(PreparedQuantizedMLPProgram)
    binding=owner._bind_abi(package)
    assert binding.argument_count==7 and len(binding.steps)==7
    assert binding.outputs==(17,) and len(binding.storage)==18
    assert binding.storage[11].shape==(3,64)
    assert binding.storage[12].shape==(3,32)
    assert binding.storage[11].storage==binding.storage[12].storage=='bf16'
    assert [list(s.inputs[:s.input_count]) for s in binding.steps]==[[0],[1,2,3],[7,9,8,10],[11],[12],[4,5,6],[13,15,14,16]]
    assert len(package.native_stages)==len(package.schedule_irs)==7


def test_full_mlp_portable_restore_without_compiler(package,monkeypatch):
    record=package.to_manifest()
    monkeypatch.setenv('TESSERA_OPT','/missing/compiler')
    monkeypatch.setenv('PATH','/missing/tools')
    restored=NativeQuantizedMLPProgram.from_manifest(record)
    assert restored.binding_digest==package.binding_digest
    assert [s.image.image_digest for s in restored.native_stages]==[s.image.image_digest for s in package.native_stages]


@pytest.mark.parametrize('mutation',['activation_bytes','activation_lifetime','second_quant_edge','down_global_root','component_graph'])
def test_resealed_mlp_shared_frame_corruption_is_rejected(package,mutation):
    plan=json.loads(package.program_json)
    if mutation=='activation_bytes':plan['buffers'][12]['bytes']*=2
    elif mutation=='activation_lifetime':plan['buffers'][12]['last_read']=3
    elif mutation=='second_quant_edge':plan['steps'][4]['inputs']=[11]
    elif mutation=='down_global_root':plan['steps'][5]['inputs'][2]=3
    else:plan['member_graphs'][4]=plan['member_graphs'][0]
    changed=replace(package,program_json=json.dumps(plan))
    changed=replace(changed,binding_digest=changed._identity())
    with pytest.raises(ValueError):changed.validate()


def test_full_mlp_rejects_recursive_portable_component(package):
    record=package.to_manifest();record['gate']=record.copy()
    with pytest.raises(ValueError,match='component schemas'):
        NativeQuantizedMLPProgram.from_manifest(record)
