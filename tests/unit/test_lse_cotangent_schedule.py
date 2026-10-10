"""Retained Graph ancestry for the explicit native LSE cotangent operand."""
from dataclasses import replace
import copy
import pytest
from tessera.compiler.graph_ir import IRArg
from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
from tests.device.nvidia.test_lse_checkpoint_native import _backward_module

pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason='requires matching native compiler')


def module(bias=False, shape=(1,2,1,3,4,4,3), causal=True):
    result=_backward_module(saved=True,bias=bias,shape=shape)
    fn=result.functions[0];op=fn.body[0]
    lse=fn.args[-1].ir_type
    fn.args.append(IRArg('row_seed',lse))
    op.operands.append('%row_seed');op.operand_types.append(str(lse))
    op.kwargs['lse_cotangent']=True
    op.kwargs['causal']=causal
    op.kwargs['scale']=1/(shape[5]**.5)
    return result


@pytest.mark.parametrize('bias',[False,True])
def test_lse_cotangent_graph_schedule_tile_ancestry(bias):
    graph=module(bias);before=copy.deepcopy(graph)
    artifact=lower_checkpoint_graph(graph,backward=True)
    artifact.validate()
    assert graph==before
    assert artifact.lse_cotangent and artifact.backward
    assert artifact.bias==bias
    assert len(artifact.names)==10+int(bias)
    assert artifact.names[6+int(bias)]=='row_seed'
    assert 'lse_cotangent = true' in artifact.schedule_ir
    assert 'lse_cotangent = true' in artifact.tile_ir
    assert 'backward_lse_output_cotangent_' in artifact.entry
    with pytest.raises(ValueError,match='binding count|metadata'):
        replace(artifact,lse_cotangent=False).validate()


@pytest.mark.parametrize('mutation',['rank','dtype','policy','missing_seed','nonboolean'])
def test_lse_cotangent_graph_rejects_inconsistent_semantics(mutation):
    graph=module();fn=graph.functions[0];op=fn.body[0]
    if mutation=='rank':
        fn.args[-1].ir_type=fn.args[1].ir_type
        op.operand_types[-1]=str(fn.args[-1].ir_type)
    elif mutation=='dtype':
        from tessera.compiler.graph_ir import IRType
        fn.args[-1].ir_type=IRType('tensor<1x2x3xf16>',('1','2','3'),'fp16')
        op.operand_types[-1]=str(fn.args[-1].ir_type)
    elif mutation=='policy':op.kwargs['lse_checkpoint']='recompute'
    elif mutation=='missing_seed':
        fn.args.pop();op.operands.pop();op.operand_types.pop()
    else:op.kwargs['lse_cotangent']='yes'
    with pytest.raises((RuntimeError,ValueError)):
        lower_checkpoint_graph(graph,backward=True)


def test_schedule_hash_binds_lse_cotangent_policy():
    artifact=lower_checkpoint_graph(module(),backward=True)
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(),artifact.schedule_ir.replace('lse_cotangent = true','lse_cotangent = false'),'--tessera-schedule-to-tile')
