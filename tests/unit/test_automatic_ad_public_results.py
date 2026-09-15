from pathlib import Path
import subprocess
import pytest
from benchmarks.record_automatic_ad_results import source
from tessera.compiler.native_public_result import materialize_ad_public_results
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.rocm_build import require_rocm_hsaco_toolkit


@pytest.fixture(autouse=True)
def _rocm_hsaco_host():
    # Every case here packages a gfx1151 HSACO: ROCm-toolkit hosts only.
    require_rocm_hsaco_toolkit()


def test_generated_ad_result_owns_shape_copy_and_capacity():
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    program=materialize_ad_public_results(source(),compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend='rocm',chip='gfx1151',capacity=4)
    metadata,_=program.validate()
    assert metadata['results']==[{'data':1,'shape':2,'capacity':4}]
    assert 'tessera.autodiff.product_pair' in program.package.arena_ir
    assert 'tessera.native_result_abi' in program.package.arena_ir


def test_generated_result_refuses_unbounded_loaded_allocation():
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    changed=source().replace('%n = arith.select %cond, %two, %four : index',
        '%integer = arith.fptosi %v : f32 to i64\n  %n = arith.index_cast %integer : i64 to index')
    with pytest.raises(subprocess.CalledProcessError):
        materialize_ad_public_results(changed,compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),
            backend='rocm',chip='gfx1151',capacity=4)



def test_tuned_ann_preserves_independent_native_schedule_provenance():
    from benchmarks.record_native_ann_execution import source as ann_source
    from tessera.compiler.native_ann import prepare_native_ann
    from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu
    from tessera.compiler.native_storage_contract import read_tensor_contract
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    logical=prepare_native_ann(ann_source(16,8),allow_reassociation=True)
    pair=materialize_native_ann_gpu(logical,compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend='rocm',chip='gfx1151',tune_transformed=True)
    pair.validate()
    assert read_tensor_contract(pair.original)['block']==[1,1,1]
    assert read_tensor_contract(pair.transformed)['block']==[16,1,1]
    assert pair.original.binding_digest!=pair.transformed.binding_digest


@pytest.mark.parametrize("rank",[2,3,4])
def test_multiple_matrix_ad_results_use_independent_shape_sidecars(rank):
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    matrix='''module { func.func @matrix(%x: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) attributes {tessera.autodiff = "reverse"} {
      %y = "tessera.mul"(%x,%x) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
      %z = "tessera.add"(%y,%x) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
      return %y,%z : tensor<2x2xf32>,tensor<2x2xf32> } }'''
    matrix=matrix.replace("2x2xf32","2x"+"1x"*(rank-2)+"2xf32")
    program=materialize_ad_public_results(matrix,compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend='rocm',chip='gfx1151',capacity=4)
    metadata,_=program.validate()
    assert metadata['results']==[{'data':1,'shape':2,'capacity':4,'rank':rank},
                                 {'data':3,'shape':4,'capacity':4,'rank':rank}]


@pytest.mark.parametrize("rank",[1,2])
def test_dynamic_backward_input_capacity_is_native_guarded(rank):
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    dynamic='''module { func.func @dynamic(%x: tensor<?xf32>) -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
      %y = "tessera.mul"(%x,%x) : (tensor<?xf32>,tensor<?xf32>) -> tensor<?xf32>
      return %y : tensor<?xf32> } }'''
    dynamic=dynamic.replace("?xf32","?x?xf32") if rank==2 else dynamic
    program=materialize_ad_public_results(dynamic,compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend='rocm',chip='gfx1151',capacity=4,input_capacity=4,role='backward')
    metadata,_=program.validate()
    assert len(metadata['inputs'])==2
    assert all(row['capacity']==4 for row in metadata['inputs'])
    assert 'arith.minui' in program.package.arena_ir
