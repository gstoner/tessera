"""Benchmark physical-ABI adapter: native and legacy images cannot be mixed."""
import ctypes as ct
from types import SimpleNamespace as NS
import numpy as np
import pytest
from benchmarks.rocm.folded_launch_arguments import folded_launch_values
from tessera.compiler.native_artifact import HAND_EMITTED_HIP_PRODUCER

def package(native, layout, pipeline):
    prov = {"native_compiler_owned": native}
    if layout is not None:
        prov["kernel_argument_layout"] = layout
    return NS(descriptor=NS(provenance=prov), image=NS(pipeline_name=pipeline))

@pytest.mark.parametrize("native", [False, True])
def test_values_match_physical_image(native):
    arrays = [np.zeros(i+1, np.uint8) for i in range(5)]
    pointers = [ct.c_void_p(0x1000+i*256) for i in range(5)]
    p = package(native, "expanded_memref" if native else None,
                "tessera-lower-to-rocm" if native else HAND_EMITTED_HIP_PRODUCER)
    values = folded_launch_values(p, pointers, arrays, (65,48,64))
    assert len(values) == (28 if native else 8)
    assert [v.value for v in values[-3:]] == [65,48,64]
    for i,pointer in enumerate(pointers):
        if native:
            assert [v.value for v in values[i*5:i*5+5]] == [pointer.value,pointer.value,0,i+1,1]
        else:
            assert values[i].value == pointer.value

@pytest.mark.parametrize("native,layout,pipeline", [
    (True,None,"tessera-lower-to-rocm"),
    (True,"raw_pointer","tessera-lower-to-rocm"),
    (True,"expanded_memref",HAND_EMITTED_HIP_PRODUCER),
    (False,None,"tessera-lower-to-rocm"),
    (False,"expanded_memref",HAND_EMITTED_HIP_PRODUCER),
])
def test_reject_crossed_contract(native,layout,pipeline):
    with pytest.raises(ValueError):
        folded_launch_values(package(native,layout,pipeline),
            [ct.c_void_p(4096)]*5,[np.zeros(1,np.uint8)]*5,(65,48,64))
