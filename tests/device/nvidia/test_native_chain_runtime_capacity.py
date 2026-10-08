"""Owning runtime proof before admitting dynamic multi-producer Graph packages."""
from contextlib import closing
import ctypes as ct
import itertools
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.unit.test_native_sm120_tensor_partition import graph

AXES=[tuple(axis for axis,bit in zip(("M","N","K"),bits,strict=True) if bit)
      for bits in itertools.product((False,True),repeat=3) if any(bits)]

@pytest.mark.parametrize("axes",AXES)
@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_checked_native_chain_capacity_before_graph_admission(axes,dtype):
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning SM120 device required")
    active={"M":17,"N":11,"K":32}
    bounds={axis:{"M":32,"N":24,"K":64}[axis] for axis in axes}
    capacities={axis:bounds.get(axis,value) for axis,value in active.items()}
    single=lhs.package_traced_lhs(graph(dtype=dtype),shape_bounds=bounds)
    # Existing single-producer Graph admission remains unchanged. This test
    # independently exercises the already exported checked append runtime ABI.
    from tessera.compiler.nvidia_tensor_lhs import _semantic_graph
    semantics={"producer":"tessera.softmax","producer_attrs":{"axis":-1},
               "consumer":"tessera.matmul","consumer_attrs":{"output_dtype":"fp32","rhs_storage_order":"row_major"},
               "roles":{"source":0,"rhs":1}}
    soft=_semantic_graph(capacities["M"],capacities["K"],capacities["N"],dtype,semantics)
    package=lhs.package_traced_lhs(soft).edge.producer
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(1251)
    with closing(PreparedLhsCall(single)) as owner:
        attach=owner.lib.tessera_nvidia_matmul_append_producer
        attach.argtypes=[ct.c_uint64,ct.c_void_p,ct.c_size_t,ct.c_char_p,ct.c_int]
        attach.restype=ct.c_int
        image=ct.create_string_buffer(package.image.payload)
        owner._check(attach(owner.handle,image,len(package.image.payload),
                            package.descriptor.entry_symbol.encode(),0))
        allocation=None
        for extents in (capacities,active,{axis:1 if axis in axes else value for axis,value in active.items()}):
            m,n,k=(extents[axis] for axis in ("M","N","K"))
            x=rng.normal(0,.2,(m,k)).astype(storage)
            rhs=rng.normal(0,.2,(k,n)).astype(storage)
            xf=x.astype(np.float64)
            norm=(xf/np.sqrt(np.mean(xf*xf,axis=-1,keepdims=True)+1e-5)).astype(storage)
            nf=norm.astype(np.float64)
            exp=np.exp(nf-nf.max(axis=-1,keepdims=True))
            edge=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
            expected=edge.astype(np.float64)@rhs.astype(np.float64)
            actual,_=owner([x,rhs])
            np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
            stats=owner.scratch_stats()
            if allocation is None:
                allocation=stats
            assert stats==allocation
        axis=axes[0]
        invalid=dict(capacities)
        invalid[axis]+=1
        m,n,k=(invalid[a] for a in ("M","N","K"))
        with pytest.raises(RuntimeError,match="outside bound"):
            owner([np.ones((m,k),storage),np.ones((k,n),storage)])
