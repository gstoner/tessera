"""Typed scaled-primal admission preserves operand roles and caller Graph."""
import copy
import numpy as np
import pytest
from tests.unit.test_public_scaled_jvp_capture import public_scaled,_inputs
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import contract,supports_typed_scaled

def module(monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    fn=ts.jit(target="rocm_gfx1201")(public_scaled)
    return fn._specialized_autodiff_module(_inputs(),{})

def test_typed_primal_contract_preserves_graph(monkeypatch):
    graph=module(monkeypatch);before=copy.deepcopy(graph).to_mlir(target="rocm_gfx1201")
    shape,fmt,names,output=contract(graph)
    assert (shape.m,shape.n,shape.k)==(17,19,256)
    assert fmt=="fp32" and names==[arg.name for arg in graph.functions[0].args]
    assert graph.to_mlir(target="rocm_gfx1201")==before

@pytest.mark.parametrize("change",["scale_shape","policy","transpose_left","batch"])
def test_typed_primal_rejects_unproved_contract(monkeypatch,change):
    graph=module(monkeypatch);op=graph.functions[0].body[0]
    if change=="scale_shape":
        from tessera.compiler.graph_ir import tensor_ir_type
        graph.functions[0].args[2].ir_type=tensor_ir_type((17,3),"fp32")
    if change=="policy":op.kwargs["numeric_policy"]={"accum":"fp32","execution_mode":"approximate"}
    if change=="transpose_left":op.kwargs["transposeA"]=True
    if change=="batch":op.kwargs["batching"]="independent_rhs"
    assert not supports_typed_scaled(graph)

def test_typed_primal_uses_operand_roles_not_argument_position(monkeypatch):
    graph=module(monkeypatch)
    fn=graph.functions[0];original=[arg.name for arg in fn.args];fn.args=fn.args[2:]+fn.args[:2]
    assert supports_typed_scaled(graph)
    assert contract(graph)[2]==original

def test_explicit_gated_tensor_annotation_preserves_status():
    from tessera.core import Tensor
    from tessera.dtype import Dtype,TesseraDtypeError
    from tessera.compiler.graph_ir import ir_args_from_signature
    with pytest.raises(TesseraDtypeError):
        Tensor["M","G","uint8"]
    byte=Dtype("uint8",allow_planned_gated=True)
    def encoded(scales:Tensor["M","G",byte]): pass
    arg=ir_args_from_signature(encoded)[0]
    assert arg.ir_type.dtype=="uint8"
    assert arg.dtype_status=="planned_gated"
    assert "ui8" in arg.to_mlir()
    assert 'tessera.dtype_status = "planned_gated"' in arg.to_mlir()
