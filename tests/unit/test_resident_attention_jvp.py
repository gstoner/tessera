"""Resident attention metadata and structural proof remain host independent."""
import itertools

import numpy as np
import pytest
import tessera as ts

from benchmarks.nvidia.benchmark_jvp_argument_order import function
from tessera.compiler.frontend_authority import certify_resident_frontends
from tessera.compiler.resident_nvidia_tensor import cuda_frontend_specs
from tests.unit.test_ordered_resident_tensor_dag import Buffer
from tests.unit.test_native_attention_jvp_runtime import contract, inputs  # noqa: F401


@pytest.mark.parametrize("order",list(itertools.permutations(("q","k","v"))))
@pytest.mark.parametrize("causal",[False,True])
def test_resident_attention_capture_and_certificate_without_host_reads(order,causal):
    values={"q":Buffer((1,2,3,4),np.float32),
            "k":Buffer((1,1,5,4),np.float32),"v":Buffer((1,1,5,3),np.float32)}
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=("q",))(function(order,causal))
    roots=tuple(values[name] for name in order)
    module=fn._specialized_autodiff_module(roots,{})
    specs=fn._resident_frontend_specs(roots)
    certificate=certify_resident_frontends(legacy_module=fn._ensure_legacy_graph_ir(),
        tracer_module=module,signature=specs,graph_consumers=("tessera.flash_attn",))
    certificate.validate()
    assert certificate.contract["concrete_executions"]==0
    assert certificate.contract["numerical_authority"]=="physical_package_required"
    assert fn._specialized_autodiff_module(roots,{}) is module


def resident_inputs():
    return tuple(Buffer(value.shape,np.float32) for value in inputs())


@pytest.mark.parametrize("index",range(4))
@pytest.mark.parametrize("field,value",[
    ("strides",(100,60,20,4)),("stream",0),("version",2),("typestr","<f2"),
    ("shape",(1,1,1,1)),("data",(0,True)),
])
def test_resident_contract_rejected_before_cuda_prepare(contract,index,field,value):
    from tessera.compiler.native_attention_jvp_runtime import execute
    roots=list(resident_inputs())
    roots[index].interface[field]=value
    with pytest.raises(ValueError):
        execute(contract,roots)


def test_mixed_roots_rejected_before_cuda_prepare(contract):
    from tessera.compiler.native_attention_jvp_runtime import execute
    roots=list(resident_inputs());roots[1]=inputs()[1]
    with pytest.raises(ValueError,match="all resident"):
        execute(contract,roots)


def test_rank_four_is_explicitly_scoped():
    value=Buffer((1,2,3,4),np.float32)
    with pytest.raises(ValueError,match="shape"):
        cuda_frontend_specs((value,))
    assert cuda_frontend_specs((value,),ranks=(4,))==(((1,2,3,4),np.dtype("float32")),)


@pytest.mark.parametrize("mutation",["policy","result","authority"])
def test_resident_certificate_refuses_frontend_drift(mutation):
    from copy import deepcopy
    roots=tuple(Buffer(shape,np.float32) for shape in
                ((1,2,3,4),(1,1,5,4),(1,1,5,3)))
    fn=ts.jit(target="nvidia_sm120")(function(("q","k","v"),False))
    module=deepcopy(fn._specialized_autodiff_module(roots,{}))
    if mutation=="policy":module.functions[0].body[0].kwargs["causal"]=True
    elif mutation=="result":module.functions[0].body[0].kwargs["return_lse"]=True
    else:module.module_attrs["tessera.frontend.authority"]='"ast"'
    with pytest.raises(ValueError):
        certify_resident_frontends(legacy_module=fn._ensure_legacy_graph_ir(),
            tracer_module=module,signature=fn._resident_frontend_specs(roots),
            graph_consumers=("tessera.flash_attn",))


def test_resident_certificate_is_content_addressed():
    from tessera.compiler.frontend_authority import ResidentFrontendCertificate
    roots=tuple(Buffer(shape,np.float32) for shape in
                ((1,2,3,4),(1,1,5,4),(1,1,5,3)))
    fn=ts.jit(target="nvidia_sm120")(function(("q","k","v"),False))
    module=fn._specialized_autodiff_module(roots,{})
    certificate=certify_resident_frontends(legacy_module=fn._ensure_legacy_graph_ir(),
        tracer_module=module,signature=fn._resident_frontend_specs(roots),
        graph_consumers=("tessera.flash_attn",))
    body=dict(certificate.contract);body["typed_graph_digest"]="0"*64
    with pytest.raises(ValueError):
        ResidentFrontendCertificate(body).validate()
