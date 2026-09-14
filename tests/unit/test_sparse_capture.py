"""Explicit JIT sparse specialization retains source output storage."""
from dataclasses import replace
import os
import numpy as np
import ml_dtypes
import pytest
from tessera.compiler.sparse_capture import compile_sparse_graph
from tessera.compiler.graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,tensor_ir_type


def graph():
    a,b,o = [tensor_ir_type(s,'fp16') for s in ((16,32),(32,16),(16,16))]
    return GraphIRModule(functions=[GraphIRFunction(name='multiply',args=[IRArg('a',a),IRArg('b',b)],result_types=[o],
        body=[IROp(op_name='tessera.matmul',result='o',operands=['%a','%b'],operand_types=[str(a),str(b)],result_type=str(o))],return_values=['%o'])])


@pytest.mark.parametrize('mutation',['epilogue','layout','shard','return'])
def test_sparse_capture_refuses_unconsumed_semantics(mutation):
    source = graph()
    if mutation == 'epilogue': source.functions[0].body[0].kwargs['activation'] = 'relu'
    if mutation == 'layout': source.functions[0].args[0].ir_type = replace(source.functions[0].args[0].ir_type,layout='col_major')
    if mutation == 'shard': source.functions[0].args[0].shard_spec = 'sharded'
    if mutation == 'return': source.functions[0].return_values = ['%a']
    with pytest.raises(ValueError): compile_sparse_graph(source)


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
@pytest.mark.parametrize("dtype",[np.float16,ml_dtypes.bfloat16])
def test_jit_sparse_capture_and_native_output_conversion(dtype,monkeypatch):
    import tessera as ts
    @ts.jit(target='rocm')
    def multiply(a,b):
        return ts.ops.matmul(a,b)
    rng = np.random.default_rng(31)
    a = (rng.normal(size=(16,32))*.2).astype(dtype)
    a.reshape(16,8,4)[:,:,2:] = 0
    b = (rng.normal(size=(32,16))*.2).astype(dtype)
    from tessera.compiler import rocm_sparse_runtime
    monkeypatch.setattr(rocm_sparse_runtime,'sparse_logical_schedule_ir',
                        lambda *a,**k: pytest.fail('Graph capture reconstructed a Python recipe'))
    package = multiply.compile_sparse_2to4(a,b)
    assert package.package.native_graph_ir == package.graph_ir
    assert 'schedule.sparse_mma' in package.package.schedule_ir
    assert package.package.output_storage in {'f16','f32','bf16'}
    result = package.run(a,b)
    assert result.dtype == (a@b).dtype
    np.testing.assert_allclose(result.astype(np.float32),(a@b).astype(np.float32),rtol=.002,atol=.0001)
    with pytest.raises(ValueError,match='identity'):
        replace(package,graph_ir=package.graph_ir+' changed').run(a,b)
    a[0,:4] = 1
    with pytest.raises(ValueError,match='invalid 2:4'):
        package.run(a,b)


def test_sparse_capture_cannot_override_policy_or_owning_target():
    with pytest.raises(ValueError,match='arithmetic'):
        compile_sparse_graph(graph(),accum='f16')
    source = graph()
    source.module_attrs['tessera.target'] = '"nvidia"'
    with pytest.raises(ValueError,match='owning'):
        compile_sparse_graph(source)


@pytest.mark.parametrize('mutation', ['transpose','module_policy','argument_policy','dtype','target'])
def test_native_sparse_graph_refuses_policy_loss(mutation):
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler required')
    source = graph()
    source.module_attrs.update({'tessera.target':'"rocm"','tessera.arch':'"gfx1201"',
                               'tessera.sparse_policy':'"checked_2to4"'})
    if mutation == 'transpose': source.functions[0].body[0].kwargs['transposeA'] = True
    ir = source.to_mlir(target='rocm',canonical=True)
    if mutation == 'transpose': assert 'transposeA = true' in ir
    if mutation == 'module_policy': ir = ir.replace('tessera.sparse_policy', 'unknown_policy = "strict", tessera.sparse_policy',1)
    if mutation == 'argument_policy': ir = ir.replace('%a: tensor<16x32xf16>','%a: tensor<16x32xf16> {tessera.layout = "col_major"}')
    if mutation == 'dtype': ir = ir.replace('f16','f32')
    if mutation == 'target': ir = ir.replace('gfx1201','gfx1151')
    with pytest.raises(RuntimeError): run_tessera_opt(tool,ir,'--tessera-graph-to-schedule')


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
def test_public_sparse_forward_keeps_logical_native_backward():
    import tessera as ts
    @ts.jit(target='rocm',autodiff='reverse',wrt=('a','b'))
    def multiply(a,b): return ts.ops.matmul(a,b)
    a=np.zeros((16,32),np.float16)
    a[:,::4]=.5
    b=np.full((32,16),.25,np.float16)
    forward=multiply.compile_sparse_2to4(a,b)
    np.testing.assert_array_equal(forward.run(a,b),a@b)
    da,db=multiply.native_backward(a,b,out_cotangents=np.ones((16,16),np.float16))
    np.testing.assert_allclose(da,np.ones((16,16),np.float32)@b.T.astype(np.float32))
    np.testing.assert_allclose(db,a.T.astype(np.float32)@np.ones((16,16),np.float32))
    assert np.all(da[a==0] != 0)
    assert multiply.last_backward_execution['evidence_target']=='rocm_gfx1201'


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
@pytest.mark.parametrize('dtype',[np.float16,ml_dtypes.bfloat16])
def test_native_automatic_sparse_dense_selection(dtype):
    import tessera as ts
    @ts.jit(target='rocm',autodiff='reverse',wrt=('a','b'))
    def multiply(a,b): return ts.ops.matmul(a,b)
    rng=np.random.default_rng(404)
    a=(rng.integers(-3,4,(32,64))*.125).astype(dtype)
    b=(rng.integers(-3,4,(64,32))*.125).astype(dtype)
    package=multiply.compile_sparse_auto(a,b)
    assert 'gpu.shuffle' in package.package.schedule_ir
    assert 'scf.if' in package.package.schedule_ir
    assert 'auto_2to4' in package.package.schedule_ir
    with pytest.raises(ValueError,match='selection disagrees'):
        replace(package.package,schedule_ir=package.package.schedule_ir.replace('auto_2to4','checked_2to4')).validate()
    # Same artifact: all dense, all sparse, then one invalid lane in one tile.
    for mode in ('dense','sparse','mixed','dense_again'):
        if mode=='sparse': a.reshape(32,16,4)[:,:,2:]=0
        if mode=='mixed': a[3,0:4]=1
        if mode=='dense_again': a[:]=.25
        expected=a.astype(np.float32)@b.astype(np.float32)
        if package.package.output_storage=='f16': expected=expected.astype(np.float16)
        np.testing.assert_allclose(package.run(a,b).astype(np.float32),expected.astype(np.float32),atol=.0001,rtol=.001)
    # Native reverse follows the unchanged logical program, not packing branches.
    if dtype==np.float16:
        da,db=multiply.native_backward(a,b,out_cotangents=np.ones((32,32),np.float16))
        np.testing.assert_allclose(da,np.ones((32,32),np.float32)@b.T.astype(np.float32),atol=.0001)
        np.testing.assert_allclose(db,a.T.astype(np.float32)@np.ones((32,32),np.float32),atol=.0001)
