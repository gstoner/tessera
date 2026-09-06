"""Trace contracts and automatic tangent ordering; numerics use the CUDA recorder."""
import numpy as np
import pytest
from benchmarks.record_jit_attention_program import function
from tessera.compiler.graph_ir import IROp
from tessera.compiler.native_attention_program import AutomaticAttentionFrame


def dense(**kwargs):
    return IROp('out','tessera.flash_attn',['%q','%k','%v'],
                ['tensor<1x2x3x4xf32>','tensor<1x1x5x4xf32>','tensor<1x1x5x3xf32>'],
                'tensor<1x2x3x3xf32>',kwargs=kwargs)


def test_dense_attention_infers_required_width_and_precise_effect():
    text=dense(causal=True).to_mlir()
    assert 'head_dim = 4 : i64' in text
    assert 'tessera.effect_kind = "pure"' in text
    assert 'head_dim = 8' in dense(head_dim=8).to_mlir()


def test_stateful_or_stochastic_attention_does_not_gain_pure_contract():
    assert 'tessera.effect_kind = "state"' in dense(dropout_p=.1).to_mlir()
    op=dense();op.operand_types[1]='!tessera.kv_cache'
    assert 'tessera.effect_kind = "state"' in op.to_mlir()
    op=dense();op.attrs='custom = true'
    assert 'tessera.effect_kind = "state"' in op.to_mlir()


def test_jit_attention_trace_preserves_requested_order_and_target(monkeypatch):
    import tessera.compiler.native_attention_program as program
    seen={}
    def compile(source,active,**kwargs):
        seen.update(source=source,active=active)
        return 'program'
    monkeypatch.setattr(program,'compile_attention_program',compile)
    values=[np.ones(shape,np.float32) for shape in ((1,2,3,4),(1,1,5,4),(1,1,5,3))]
    assert function(('k','q')).compile_native_attention_jvp(*values,compiler='unused',llvm_bin='unused')=='program'
    assert seen['active']==(1,0)
    assert 'tessera.target = "nvidia_sm120"' in seen['source']
    assert 'tessera.arch = "sm_120"' in seen['source']
    assert 'head_dim = 4 : i64' in seen['source']
    assert 'tessera.effect_kind = "pure"' in seen['source']


def test_automatic_frame_maps_tangents_by_requested_order():
    class Frame:
        def jvp(self,*values): return values
    frame=AutomaticAttentionFrame(Frame(),(1,0),{2:'inactive'})
    assert frame.jvp('dk','dq')==('dq','dk','inactive')
    with pytest.raises(ValueError,match='arity'):
        frame.jvp('dk')
