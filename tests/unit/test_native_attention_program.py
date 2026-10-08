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


def test_reverse_jit_attention_trace_preserves_gradient_order(monkeypatch):
    import tessera as ts
    import tessera.compiler.native_attention_program as program
    seen = {}
    def compile(source, active, **kwargs):
        seen.update(source=source, active=active)
        return "reverse-program"
    monkeypatch.setattr(program, "compile_attention_vjp_program", compile)
    @ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("k", "q"))
    def attention(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)
    values = [np.ones(shape,np.float32) for shape in ((1,2,3,4),(1,1,5,4),(1,1,5,3))]
    assert attention.compile_native_attention_vjp(*values, compiler="unused") == "reverse-program"
    assert seen["active"] == (1, 0)
    assert 'tessera.autodiff = "reverse"' in seen["source"]
    assert 'tessera.target = "nvidia_sm120"' in seen["source"]


def test_reverse_frame_selects_gradients_without_recomputing():
    from tessera.compiler.native_attention_program import AutomaticAttentionVJPFrame
    class Frame:
        def backward(self, cotangent):
            assert cotangent == "seed"
            return ("dq", "dk", "dv")
    assert AutomaticAttentionVJPFrame(Frame(), (1, 0)).backward("seed") == ("dk", "dq")

def test_reverse_bias_trace_preserves_operand_and_requested_order(monkeypatch):
    from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import function
    import tessera.compiler.native_attention_program as program
    seen = {}
    def compile(source, active, **kwargs):
        seen.update(source=source, active=active)
        return "biased-program"
    monkeypatch.setattr(program, "compile_attention_vjp_program", compile)
    values = [np.ones(shape,np.float32) for shape in
        ((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,3,5))]
    assert function(("bias","k"), True).compile_native_attention_vjp(*values,compiler="unused") == "biased-program"
    assert seen["active"] == (3,1)
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 1>" in seen["source"]
    assert 'tessera.effect_kind = "pure"' in seen["source"]
    assert 'attn_bias = "%' not in seen["source"]


def test_reverse_bias_frame_selects_fourth_native_result():
    from tessera.compiler.native_attention_program import AutomaticAttentionVJPFrame
    class Frame:
        def backward(self, seed):
            return ("dq","dk","dv","dbias")
    assert AutomaticAttentionVJPFrame(Frame(),(3,1)).backward("seed") == ("dbias","dk")

def test_reverse_capture_maps_frontend_inputs_to_native_roles():
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    class Pair:
        def capture(self, q, k, v, *, bias=None, asynchronous=False):
            assert (q,k,v,bias) == ("q","k","v","bias")
            return "frame"
    program = NativeAttentionVJPProgram(Pair(), (3,0), (3,0,2,1))
    assert program.capture("k","bias","v","q")._frame == "frame"
    assert program.capture("k","v","q",bias="bias")._frame == "frame"
    with pytest.raises(ValueError, match="arity"):
        program.capture("k","bias","v")

def test_forward_capture_maps_frontend_inputs_to_native_roles(monkeypatch):
    from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
    class Pair:
        def capture(self, q, k, v):
            assert (q,k,v) == ("q","k","v")
            raise RuntimeError("canonical capture reached")
    monkeypatch.setattr(NativeAttentionJVPProgram,"validate",lambda self:None)
    program=NativeAttentionJVPProgram(Pair(),None,(1,2),(1,2,0),("v","q","k"))
    with pytest.raises(RuntimeError,match="canonical capture"):
        program.capture("v","q","k")
    with pytest.raises(RuntimeError,match="canonical capture"):
        program.capture(q="q",k="k",v="v")
    with pytest.raises(RuntimeError,match="canonical capture"):
        program.capture("v",q="q",k="k")

@pytest.mark.parametrize("mapping", [(0,0,2),(0,1),(0,1,3),(False,1,2)])
def test_forward_capture_refuses_nonpermutation_before_allocation(mapping):
    from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
    class Pair:
        def capture(self,*args):
            raise AssertionError("allocated before checking native mapping")
    with pytest.raises(ValueError,match="permutation"):
        NativeAttentionJVPProgram(Pair(),None,(0,),mapping).capture("q","k","v")

def test_forward_capture_uses_signature_names_before_physical_role_mapping(monkeypatch):
    from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
    class Pair:
        def capture(self, q, k, v):
            assert (q,k,v)==("query","physical-key","physical-value")
            raise RuntimeError("mapped capture reached")
    monkeypatch.setattr(NativeAttentionJVPProgram,"validate",lambda self:None)
    program=NativeAttentionJVPProgram(Pair(),None,(2,),(0,2,1),("q","k","v"))
    with pytest.raises(RuntimeError,match="mapped capture"):
        program.capture(q="query",k="physical-value",v="physical-key")
    for args,kwargs in [
        (("query",),{"q":"duplicate","k":"physical-value","v":"physical-key"}),
        ((),{"q":"query","k":"physical-value"}),
        ((),{"q":"query","k":"physical-value","v":"physical-key","extra":"bad"}),
    ]:
        with pytest.raises(TypeError):
            program.capture(*args,**kwargs)


@pytest.mark.parametrize("key,value", [
    ("bias", 1), ("bias", None), ("bias_gradient", "false"),
])
def test_native_program_rejects_nonboolean_policy(key, value):
    from tessera.compiler.native_attention_program import _policy_bool
    with pytest.raises(ValueError, match="boolean policy"):
        _policy_bool({key: value}, key)


@pytest.mark.parametrize("value", [None, "012", [0, True], [0, 1.0]])
def test_native_program_rejects_noninteger_sequence_policy(value):
    from tessera.compiler.native_attention_program import _policy_indices
    with pytest.raises(ValueError, match="integer sequence"):
        _policy_indices({"frontend_argument_indices": value}, "frontend_argument_indices")
