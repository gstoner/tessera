from copy import deepcopy
import numpy as np
import pytest
import tessera as ts
from tessera.compiler import scheduled_matmul,nvidia_native
from tessera.compiler.nvidia_tensor_rhs import package_traced_rmsnorm_rhs

pytestmark=pytest.mark.skipif(scheduled_matmul.find_tessera_opt() is None or not nvidia_native.tools_available(),reason="native compiler required")


@ts.jit(target="nvidia_sm120")
def rhs_product(lhs,source):
    return ts.ops.matmul(lhs,ts.ops.rmsnorm(source,eps=1e-5),output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def rhs_reordered(source,lhs):
    return ts.ops.matmul(lhs,ts.ops.rmsnorm(source,eps=1e-5),output_dtype="fp32")


@pytest.mark.parametrize("function",[rhs_product,rhs_reordered])
def test_frontend_rhs_trace_packages_in_argument_order(function):
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    args=(a,x) if function is rhs_product else (x,a)
    program=function.compile_native_rhs_matmul(*args)
    assert program.argument_names==tuple(function.arg_names)
    assert program.argument_names[program.source_index]=="source"
    assert program.argument_names[program.lhs_index]=="lhs"
    assert "tessera.rmsnorm" in program.graph_ir
    program.edge.validate()
    with pytest.raises(TypeError):
        program.execute_resident(*args,source=x)


def test_trace_refuses_different_edge_without_mutation():
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    module=rhs_product._traced_autodiff_module((a,x),{})
    changed=deepcopy(module)
    changed.functions[0].body[1].operands.reverse()
    before=changed.to_mlir(verify=False)
    with pytest.raises(ValueError,match="CFG|matmul|K dimension"):
        package_traced_rmsnorm_rhs(changed,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert changed.to_mlir(verify=False)==before


def test_trace_rejects_cfg_effect_drift_before_partition():
    from dataclasses import replace
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    changed=deepcopy(rhs_product._traced_autodiff_module((a,x),{}))
    fn=changed.functions[0]
    cfg=fn.structured_cfg
    entry=cfg.blocks[0]
    operations=(replace(entry.operations[0],effect="write"),*entry.operations[1:])
    fn.structured_cfg=replace(cfg,blocks=(replace(entry,operations=operations),*cfg.blocks[1:]))
    with pytest.raises(ValueError,match="CFG"):
        package_traced_rmsnorm_rhs(changed,pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("attribute,value",[("gamma",0.0),("gamma",2.0),("axis",0)])
def test_rhs_trace_does_not_drop_unsupported_norm_semantics(attribute,value):
    a=np.zeros((17,35),np.float16)
    x=np.ones((35,19),np.float16)
    changed=deepcopy(rhs_product._traced_autodiff_module((a,x),{}))
    changed.functions[0].body[0].kwargs[attribute]=value
    with pytest.raises(ValueError,match="gamma|axis"):
        package_traced_rmsnorm_rhs(changed,pipeline_name="tessera-nvidia-pipeline-sm120")
