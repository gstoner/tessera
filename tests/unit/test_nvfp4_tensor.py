"""Logical storage validation and value-free frontend tracing."""
import numpy as np
import pytest
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tessera.compiler.trace import trace, to_graph_ir_module
import tessera as ts

def product(a, b, sa, sb):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum":"fp32", "execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block", "block":[1,16], "format":"ue4m3"})

def test_trace_retains_logical_dimensions_without_decoding_storage():
    a=NVFP4Tensor(np.zeros((3,4),np.uint8),(3,7),1)
    b=NVFP4Tensor(np.zeros((4,5),np.uint8),(7,5),0)
    traced=trace(product,a,b,np.ones((3,1),np.uint8),np.ones((1,5),np.uint8))
    module=to_graph_ir_module(traced,name="product",source_hash="a"*64,target="nvidia_sm120")
    assert str(module.functions[0].args[0].ir_type)=="tensor<3x7x!tessera.nvfp4>"
    assert str(module.functions[0].result_types[0])=="tensor<3x5xf32>"
    with pytest.raises(TypeError,match="not an ordinary"):
        np.asarray(a)

@pytest.mark.parametrize("shape,axis",[((0,7),1),((3,7),True),((3,7),-1),((3,7),2),((3,7,1,1),1)])
def test_invalid_logical_storage_refused(shape,axis):
    with pytest.raises(ValueError):
        NVFP4Tensor(np.zeros((3,4),np.uint8),shape,axis)

@pytest.mark.parametrize("storage",[np.zeros((3,4),np.float32),np.zeros((3,5),np.uint8),np.zeros((3,8),np.uint8)[:,::2]])
def test_invalid_physical_storage_refused(storage):
    with pytest.raises(ValueError):
        NVFP4Tensor(storage,(3,7),1)

def test_revalidation_catches_mutated_caller_buffer():
    storage=np.zeros((3,4),np.uint8)
    binding=NVFP4Tensor(storage,(3,7),1)
    storage.shape=(2,6)
    with pytest.raises(ValueError,match="shape differs"):
        binding.validate()
