"""Host-free CUDA root admission; no device or numerical backend is invoked."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.compiler.nvidia_tensor_dag import _checked_device_arguments, execute_resident
from tessera.compiler.resident_nvidia_tensor import ordered_resident_views


class Buffer:
    def __init__(self, shape, dtype=np.float16, stream=31):
        self.dtype=np.dtype(dtype)
        self.interface={"shape":shape,"typestr":self.dtype.str,"strides":None,
                        "data":(4096,True),"version":3,"stream":stream}

    @property
    def __cuda_array_interface__(self):
        return deepcopy(self.interface)

    def __array__(self,*args,**kwargs):
        pytest.fail("CUDA root coerced to a host array")


def program(dynamic=False):
    return SimpleNamespace(argument_names=("a","b"),
        semantics={"roles":{"source":0,"rhs":1}},
        edge=SimpleNamespace(dtype="fp16",m=32 if dynamic else 17,
                             n=24 if dynamic else 11,k=64 if dynamic else 32,
                             dynamic_m=dynamic,dynamic_n=dynamic,dynamic_k=dynamic))


@pytest.mark.parametrize("streams",[(31,47),(31,31),(1,2)])
@pytest.mark.parametrize("dynamic",[False,True])
def test_external_roots_are_metadata_only_and_keep_borrowed_owners(streams,dynamic):
    a,b=Buffer((17,32),stream=streams[0]),Buffer((32,11),stream=streams[1])
    values,shape=_checked_device_arguments(program(dynamic),[a,b])
    assert values==[a,b] and shape==(17,11)
    out=Buffer((17,11),np.float32,stream=73)
    out.interface["data"]=(8192,False)
    views,declared=ordered_resident_views([a,b,out],73,writable_from=2)
    assert declared==streams
    assert views[0].data==4096 and views[2].data==8192
    assert views[0].bytes==17*32*2


@pytest.mark.parametrize("field,value",[
    ("version",2),("shape",(0,32)),("shape",(True,32)),("shape",(2**62,32)),
    ("shape",(17,33)),("shape",(17,32,1)),("typestr","<u2"),("typestr","not_a_dtype"),
    ("strides",(66,2)),("strides",(-64,2)),("data",(0,False)),
    ("data",(4096,0)),("stream",None),("stream",0),("stream",False),
    ("stream",-1),("stream",2**64),
])
def test_malformed_root_rejected_before_native_owner(field,value,monkeypatch):
    from tessera.compiler import prepared_nvidia_lhs
    def forbidden(*args,**kwargs):pytest.fail("malformed CUDA root reached native prepare")
    monkeypatch.setattr(prepared_nvidia_lhs,"PreparedLhsCall",forbidden)
    a,b=Buffer((17,32)),Buffer((32,11))
    a.interface[field]=value
    with pytest.raises(ValueError):
        execute_resident(program(),(a,b),{})


def test_mixed_host_and_cuda_roots_rejected_before_native_owner(monkeypatch):
    from tessera.compiler import prepared_nvidia_lhs
    def forbidden(*args,**kwargs):pytest.fail("mixed roots reached native prepare")
    monkeypatch.setattr(prepared_nvidia_lhs,"PreparedLhsCall",forbidden)
    with pytest.raises(ValueError,match="all roots resident"):
        execute_resident(program(),(Buffer((17,32)),np.zeros((32,11),np.float16)),{})


def test_result_stream_still_requires_exact_launch_ownership():
    a,b=Buffer((17,32)),Buffer((32,11))
    output=Buffer((17,11),np.float32,stream=74)
    output.interface["data"]=(8192,False)
    with pytest.raises(RuntimeError,match="must match"):
        ordered_resident_views([a,b,output],73,writable_from=2)


def test_fused_root_metadata_carries_bias_and_residual():
    p=program(True);p.argument_names=("a","b","bias","residual")
    p.semantics["roles"].update(bias=2,residual=3)
    values=[Buffer((17,32)),Buffer((32,11)),Buffer((11,),np.float32),
            Buffer((17,11),np.float32)]
    selected,shape=_checked_device_arguments(p,values)
    assert selected==values and shape==(17,11)
    values[2]=Buffer((12,),np.float32)
    with pytest.raises(ValueError,match="bias shape"):
        _checked_device_arguments(p,values)


def test_standard_typestr_does_not_require_numpy_compatible_provider_dtype():
    a,b=Buffer((17,32)),Buffer((32,11))
    # Provider-specific dtype objects (e.g. torch.float16) are outside NumPy's
    # dtype parser. Primitive CAI metadata carries the storage contract.
    a.dtype="provider.float16"
    b.dtype=object()
    values,shape=_checked_device_arguments(program(),[a,b])
    assert values==[a,b] and shape==(17,11)
    output=Buffer((17,11),np.float32,stream=73)
    output.dtype=object()
    output.interface["data"]=(8192,False)
    views,_=ordered_resident_views([a,b,output],73,writable_from=2)
    assert [view.dtype for view in views]==[2,2,1]


def test_ordered_output_must_be_writable():
    a,b=Buffer((17,32)),Buffer((32,11))
    output=Buffer((17,11),np.float32,stream=73)
    with pytest.raises(ValueError,match="writable output"):
        ordered_resident_views([a,b,output],73,writable_from=2)
