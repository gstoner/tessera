from dataclasses import replace
import numpy as np
import pytest
from tessera.compiler.ann_fragment import ParameterSnapshot, extract_ann_fragment, validate_affine_composition
from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IRType, IROp
from tessera.compiler import evaluator


def graph(weights, biases, activation=False):
    def ty(shape):
        return 'tensor<' + 'x'.join(map(str,shape)) + 'xf32>'
    args = [IRArg('%x',IRType(ty((4,weights[0].shape[0]))))]
    ops, parameters = [], {}
    current = '%x'
    for i,(w,b) in enumerate(zip(weights,biases,strict=True)):
        wt,bt,out = ty(w.shape),ty(b.shape),ty((4,w.shape[1]))
        args += [IRArg(f'%w{i}',IRType(wt)),IRArg(f'%b{i}',IRType(bt))]
        parameters[f'%w{i}'] = ParameterSnapshot.capture(f'w{i}',w)
        parameters[f'%b{i}'] = ParameterSnapshot.capture(f'b{i}',b)
        ops += [IROp(f'%m{i}','tessera.matmul',[current,f'%w{i}'],[ty((4,w.shape[0])),wt],out),
                IROp(f'%a{i}','tessera.add',[f'%m{i}',f'%b{i}'],[out,bt],out)]
        current = f'%a{i}'
        if activation:
            ops.append(IROp(f'%r{i}','tessera.relu',[current],[out],out))
            current = f'%r{i}'
    return GraphIRFunction('ann',args,[IRType(out)],ops,return_values=[current]),parameters


def fragments():
    w=np.arange(6,dtype=np.float32).reshape(2,3)/8
    b=np.arange(3,dtype=np.float32)/4
    v=np.arange(6,dtype=np.float32).reshape(3,2)/16
    c=np.ones(2,np.float32)
    original, params=graph([w,v],[b,c])
    fused, fused_params=graph([w@v],[b@v+c])
    return extract_ann_fragment(original,params),extract_ann_fragment(fused,fused_params)


def test_inventory_and_composition():
    before,after=fragments()
    assert before.dense_slots==17 and after.dense_slots==6
    assert validate_affine_composition(before,after,allow_reassociation=True)['after_dense_slots']==6
    with pytest.raises(ValueError,match='reassociation'):
        validate_affine_composition(before,after)


def test_snapshot_is_owned():
    values=np.ones(3,np.float32)
    saved=ParameterSnapshot.capture('bias',values)
    values[:]=7
    np.testing.assert_array_equal(saved.array(),1)
    assert not saved.array().flags.writeable


@pytest.mark.parametrize('mutation',['bias','activation','shape'])
def test_mutation_refuses_candidate_before_execution(monkeypatch,mutation):
    before,after=fragments()
    layer=after.layers[0]
    if mutation=='bias':
        layer=replace(layer,bias=ParameterSnapshot.capture('bad',layer.bias.array()+1))
    elif mutation=='shape':
        layer=replace(layer,weight=ParameterSnapshot.capture('bad',np.zeros((3,2),np.float32)))
    else:
        layer=replace(layer,activation='relu')
    after=replace(after,layers=(layer,))
    monkeypatch.setattr(evaluator,'run_native',lambda *args: pytest.fail('invalid candidate executed'))
    with pytest.raises(ValueError):
        evaluator.ann_composition_equivalence('rocm',None,None,(),(),before=before,after=after,allow_reassociation=True)


def test_native_boundary_not_replaced_by_algebra(monkeypatch):
    before,after=fragments()
    monkeypatch.setattr(evaluator,'run_native',lambda *args:(None,False))
    assert evaluator.ann_composition_equivalence('rocm',None,None,(),(),before=before,after=after,allow_reassociation=True).relation=='inconclusive'


def test_unknown_parameter_and_policy_refused():
    fn,p=graph([np.ones((2,2),np.float32)],[np.ones(2,np.float32)])
    with pytest.raises(ValueError,match='ownership'):
        extract_ann_fragment(fn,{})
    fn.body[0].attrs='numeric_policy = "strict"'
    with pytest.raises(ValueError,match='policy'):
        extract_ann_fragment(fn,p)


def test_shared_storage_counted_once():
    fn,p=graph([np.eye(2,dtype=np.float32)]*2,[np.ones(2,np.float32)]*2)
    p['%w1']=p['%w0']
    p['%b1']=p['%b0']
    fragment=extract_ann_fragment(fn,p)
    assert fragment.dense_slots==12 and fragment.unique_storage_slots==6


def test_graph_argument_names_use_canonical_ssa_bindings():
    fn,p=graph([np.ones((2,2),np.float32)],[np.ones(2,np.float32)])
    for argument in fn.args:
        argument.name=argument.name.lstrip('%')
    assert extract_ann_fragment(fn,{name.lstrip('%'):value for name,value in p.items()}).dense_slots==6


def test_frozen_parameters_are_not_trainable_slots():
    fn,p=graph([np.ones((2,2),np.float32)],[np.ones(2,np.float32)])
    p['%w0']=replace(p['%w0'],trainable=False)
    fragment=extract_ann_fragment(fn,p)
    assert fragment.dense_slots==6 and fragment.unique_storage_slots==6
    assert fragment.unique_trainable_slots==2
