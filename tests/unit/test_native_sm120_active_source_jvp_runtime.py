"""Public source-active artifact replay, native-owner reuse and witness drift."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.portable_nvidia_tensor_jvp import digest
from tessera.compiler.portable_nvidia_active_source import restore
from benchmarks.nvidia.active_source_jvp_reference import inputs,oracle,product,product_reversed







@pytest.fixture(scope='module')
def owning_gpu():
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    if find_tessera_opt() is None or rt._nvidia_device_name() != 'sm_120':
        pytest.skip('requires owning RTX5070 and matching native compiler')


@pytest.fixture(scope='module')
def host_package():
    path=Path(__file__).resolve().parents[1]/'fixtures/native_sm120/active_source_jvp_f16.json'
    artifact=rt.RuntimeArtifact.from_json(path.read_text())
    plan=json.loads(artifact.metadata['native_program']['plan_json'])
    roots=tuple(np.zeros(buffer['shape'],np.float16) for buffer in plan['buffers'][:3])
    return None,roots,artifact,None,None,'f16',False


@pytest.fixture(scope='module',params=[('f16',False),('f16',True),('bf16',False),('bf16',True)])
def package(request,owning_gpu):
    storage,permuted=request.param
    a,b,da=inputs({'shape_mnk':(17,19,35),'storage':storage})
    args=(b,a) if permuted else (a,b)
    fn=ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('a',))(
        product_reversed if permuted else product)
    pair=fn.native_jvp(*args,tangents=(da,))
    artifact=fn.native_jvp_runtime_artifact()
    expected,budgets=oracle(a,b,da)
    check(pair,expected,budgets)
    yield fn,(*args,da),artifact,expected,budgets,storage,permuted
    fn.close_native_storage()


def check(outputs,expected,budgets):
    assert len(outputs)==2 and not np.shares_memory(*outputs)
    for actual,reference,budget in zip(outputs,expected,budgets,strict=True):
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        assert np.all(np.abs(actual.astype(np.float64)-reference)<=budget)


def reseal(data):
    for record in data['members']:
        record['member_digest']=digest({k:v for k,v in record.items() if k!='member_digest'})
    data['contract_digest']=digest({k:v for k,v in data.items() if k!='contract_digest'})


def test_json_public_replay_borrows_live_jit_lease(package,monkeypatch):
    fn,roots,artifact,expected,budgets,_,permuted=package
    import tessera.compiler.prepared_nvidia_active_source as prepared
    from tessera.compiler import nvidia_native
    owner=next(iter(fn._native_prepared_jvp_calls.values()))
    handle=owner.handle;stats=owner.scratch_stats()
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    names=['primal_0','primal_1',f'tangent_{int(permuted)}']
    assert restored.metadata['arg_names']==names
    with monkeypatch.context() as warm:
        warm.setattr(prepared.PreparedActiveSourceJVP,'from_manifest',classmethod(lambda *a,**k:pytest.fail('live lease prepared twice')))
        warm.setattr(nvidia_native,'_compile_tile_ir',lambda *a,**k:pytest.fail('replay compiled'))
        for args in (roots,dict(zip(names,roots,strict=True))):
            result=rt.launch(restored,args)
            assert result['ok'],result
            assert result['execution_kind']=='native_gpu'
            assert result['compiler_path']=='nvidia_sm120_active_source_jvp_compiled'
            assert len(result['component_receipts'])==3
            assert [r['abi']['kind'] for r in result['component_receipts']]==['row_pair','matmul','matmul']
            check(result['output'],expected,budgets)
        check(fn.native_jvp(*roots[:2],tangents=(roots[2],)),expected,budgets)
    assert owner.handle==handle and owner.scratch_stats()==stats and owner._finalizer.alive


def corruptions():
    yield 'schema',lambda d:d.update(schema='tessera.nvidia.tensor_rhs_jvp.v1')
    yield 'member-count',lambda d:d['members'].pop()
    yield 'image-digest',lambda d:d['members'][0].update(image_digest='0'*64)
    yield 'image-target',lambda d:d['members'][0].update(ptx=d['members'][0]['ptx'].replace('.target sm_120a','.target sm_90a'))
    yield 'entry',lambda d:d['members'][0].update(entry='another_kernel')
    yield 'step-size',lambda d:d['members'][0]['abi'].update(step_bytes=64)
    yield 'output-arity',lambda d:d['members'][0]['abi'].update(outputs=[3])
    yield 'seed-role',lambda d:d['members'][0]['abi'].update(inputs=[0,1])
    yield 'dimensions',lambda d:d['members'][1]['abi'].update(dimensions=[17,19,36])
    yield 'storage',lambda d:d['members'][0]['abi'].update(storage='f32')
    yield 'block',lambda d:d['members'][0]['abi'].update(block_size=32)
    yield 'parameters',lambda d:d['members'][0]['abi'].update(parameters=['ptr']*6)
    yield 'resource',lambda d:d['members'][0]['resources'].update(registers_per_thread=-1)
    yield 'extra-field',lambda d:d['members'][0].update(already_verified=True)
    yield 'member-graph',lambda d:d['members'][0].update(graph_ir=d['members'][0]['graph_ir']+'\n// changed')
    yield 'native-witness',lambda d:d['members'][0].update(native_ir=d['members'][0]['native_ir']+'\n// changed')
    yield 'source-provenance',lambda d:d['members'][0]['provenance'].update(source_graph_ir_digest='0'*64)
    def change_plan(data,key,value):
        plan=json.loads(data['plan_json']);plan[key]=value;data['plan_json']=json.dumps(plan)
    yield 'plan-roles',lambda d:change_plan(d,'role_indices',[1,0,2])
    yield 'plan-state',lambda d:change_plan(d,'execution_state','native_gpu')
    yield 'plan-source',lambda d:change_plan(d,'source_graph_ir','module {}')
    def lifetime(data):
        plan=json.loads(data['plan_json']);plan['buffers'][3]['last_read']=0
        data['plan_json']=json.dumps(plan)
    yield 'lifetime',lifetime


@pytest.mark.parametrize('name,mutate',list(corruptions()),ids=lambda v:v if isinstance(v,str) else '')
def test_resealed_inconsistent_artifacts_reject_before_provider(host_package,name,mutate,monkeypatch):
    _,roots,artifact,_,_,_,_=host_package
    data=copy.deepcopy(artifact.metadata['native_program'])
    mutate(data)
    reseal(data)
    metadata=copy.deepcopy(artifact.metadata);metadata['native_program']=data
    bad=rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=metadata)
    monkeypatch.setattr(rt,'_load_nvidia_ptx_launch',lambda:pytest.fail('provider accessed for invalid artifact'))
    result=rt.launch(bad,roots)
    assert not result['ok'],(name,result)


@pytest.mark.parametrize('change',['parent-graph','arg-names','target','external-stream'])
def test_parent_binding_rejects_before_provider(host_package,change,monkeypatch):
    _,roots,artifact,_,_,_,_=host_package
    metadata=copy.deepcopy(artifact.metadata);graph=artifact.graph_ir
    stream=None
    if change=='parent-graph':graph='module {}'
    if change=='arg-names':metadata['arg_names']=['x','y','z']
    if change=='target':metadata['target']='rocm_gfx1201'
    if change=='external-stream':stream=object()
    bad=rt.RuntimeArtifact(graph_ir=graph,metadata=metadata)
    monkeypatch.setattr(rt,'_load_nvidia_ptx_launch',lambda:pytest.fail('provider accessed for invalid parent'))
    assert not rt.launch(bad,roots,stream=stream)['ok']


def test_restore_validation_is_compiler_and_provider_free(host_package,monkeypatch):
    _,_,artifact,_,_,_,_=host_package
    monkeypatch.setattr(rt,'_load_nvidia_ptx_launch',lambda:pytest.fail('provider used by restore'))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('compiler used by restore'))
    native,_,members=restore(artifact.metadata['native_program'])
    assert native.tool==Path('/unavailable/compiler') and len(members)==3
    assert native.manifest['execution_state']=='artifact_only'


CHILD='''
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler import nvidia_native,native_sm120_active_source
folder=Path(sys.argv[1]);storage=sys.argv[2]
dtype=np.float16 if storage=='f16' else ml_dtypes.bfloat16
artifact=rt.RuntimeArtifact.from_json((folder/'artifact.json').read_text())
values=tuple(np.load(folder/f'root-{i}.npy').view(dtype) for i in range(3))
def forbidden(*a,**k):raise AssertionError('compiler or image file access during portable replay')
subprocess.run=forbidden
Path.read_bytes=forbidden
nvidia_native._compile_tile_ir=forbidden
native_sm120_active_source.export_native_active_source_jvp=forbidden
native_sm120_active_source.NativeActiveSourceProgram.project_member=forbidden
native_sm120_active_source.NativeActiveSourceProgram.lower_members=forbidden
os.environ['PATH']='/unavailable'
os.environ['TESSERA_OPT']='/unavailable/compiler'
os.environ['TESSERA_NVIDIA_OPT']='/unavailable/compiler'
for mode in ('positional','named'):
 args=values if mode=='positional' else dict(zip(artifact.metadata['arg_names'],values,strict=True))
 result=rt.launch(artifact,args)
 assert result['ok'],result
 assert result['execution_kind']=='native_gpu' and len(result['component_receipts'])==3
 for i,value in enumerate(result['output']):np.save(folder/f'{mode}-{i}.npy',value)
print(json.dumps({'ok':True,'target':'nvidia_sm120','components':3}))
'''


def test_fresh_process_replay_without_compiler(package,tmp_path):
    _,roots,artifact,expected,budgets,storage,_=package
    (tmp_path/'artifact.json').write_text(artifact.to_json())
    for i,root in enumerate(roots):np.save(tmp_path/f'root-{i}.npy',root.view(np.uint16))
    environment=os.environ.copy()
    environment.update(PATH='/unavailable',TESSERA_OPT='/unavailable/compiler',
                       TESSERA_NVIDIA_OPT='/unavailable/compiler',TESSERA_LLVM_BIN='/unavailable')
    result=subprocess.run([sys.executable,'-c',CHILD,str(tmp_path),storage],
                          capture_output=True,text=True,env=environment)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.strip())['ok']
    for mode in ('positional','named'):
        check(tuple(np.load(tmp_path/f'{mode}-{i}.npy') for i in range(2)),expected,budgets)


def test_replay_consumes_new_root_values(package):
    _,roots,artifact,_,_,_,permuted=package
    changed=[a.copy() for a in roots]
    changed[int(permuted)][:]*=0.75
    changed[1-int(permuted)][:]*=-0.5
    changed[2][:]+=0.03125
    a=changed[int(permuted)];b=changed[1-int(permuted)];da=changed[2]
    expected,budgets=oracle(a,b,da)
    result=rt.launch(artifact,changed)
    assert result['ok'],result
    check(result['output'],expected,budgets)


def test_receipt_mutation_does_not_modify_saved_package(package):
    _,roots,artifact,expected,budgets,_,_=package
    snapshot=copy.deepcopy(artifact.metadata['native_program'])
    result=rt.launch(artifact,roots)
    assert result['ok'],result
    result['component_receipts'][0]['abi']['inputs'][0]=99
    assert artifact.metadata['native_program']==snapshot
    result=rt.launch(artifact,roots)
    assert result['ok'],result
    check(result['output'],expected,budgets)


def test_cached_certificate_is_bound_to_entire_payload(package,monkeypatch):
    _,roots,artifact,_,_,_,_=package
    import tessera.compiler.nvidia_tensor_jvp_runtime as binding
    binding._certificates.clear()
    original=binding.restore_active;calls=[]
    def counted(data):
        calls.append(1)
        return original(data)
    monkeypatch.setattr(binding,'restore_active',counted)
    for _ in range(2):assert rt.launch(artifact,roots)['ok']
    assert len(calls)==1
    metadata=copy.deepcopy(artifact.metadata)
    metadata['native_program']['members'][0]['abi']['inputs'][0]=99
    # Preserve the display digest; a full-payload cache must still reject.
    bad=rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=metadata)
    monkeypatch.setattr(rt,'_load_nvidia_ptx_launch',lambda:pytest.fail('bad cached artifact reached provider'))
    assert not rt.launch(bad,roots)['ok']
    assert len(calls)==2


@pytest.mark.parametrize('storage',['f16','bf16'])
@pytest.mark.parametrize('permuted',[False,True])
@pytest.mark.parametrize('shape',[(1,1,32),(129,33,257)])
def test_public_edge_and_ragged_shapes(storage,permuted,shape,owning_gpu):
    a,b,da=inputs({'shape_mnk':shape,'storage':storage})
    args=(b,a) if permuted else (a,b)
    expected,budgets=oracle(a,b,da)
    fn=ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('a',))(
        product_reversed if permuted else product)
    try:
        check(fn.native_jvp(*args,tangents=(da,)),expected,budgets)
        for module in fn._autodiff_specializations.values():
            assert module.module_attrs['tessera.frontend.authority']=='"tracer"'
        artifact=fn.native_jvp_runtime_artifact()
        fn.close_native_storage()
        result=rt.launch(artifact,(*args,da))
        assert result['ok'],result
        check(result['output'],expected,budgets)
    finally:
        fn.close_native_storage()


def test_every_source_active_abi_export_is_in_the_inventory():
    from tessera.compiler.runtime_abi_audit import collect_runtime_abi
    expected={'tessera_nvidia_active_source_program_'+name for name in
              ('prepare','invoke','profile','scratch_stats','close')}
    actual={symbol.name for symbol in collect_runtime_abi() if symbol.backend=='nvidia'}
    assert expected <= actual


def _read_producer_storage(record,source,seed):
    """Test-only direct image execution exposes both typed producer edges."""
    import ctypes as ct
    driver=ct.CDLL('libcuda.so.1')
    signatures={
        'cuInit':[ct.c_uint], 'cuDeviceGet':[ct.POINTER(ct.c_int),ct.c_int],
        'cuCtxGetCurrent':[ct.POINTER(ct.c_void_p)], 'cuCtxSetCurrent':[ct.c_void_p],
        'cuDevicePrimaryCtxRetain':[ct.POINTER(ct.c_void_p),ct.c_int],
        'cuDevicePrimaryCtxRelease_v2':[ct.c_int], 'cuCtxSynchronize':[],
        'cuModuleLoadData':[ct.POINTER(ct.c_void_p),ct.c_void_p],
        'cuModuleGetFunction':[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],
        'cuModuleUnload':[ct.c_void_p], 'cuMemAlloc_v2':[ct.POINTER(ct.c_uint64),ct.c_size_t],
        'cuMemFree_v2':[ct.c_uint64],
        'cuMemcpyHtoD_v2':[ct.c_uint64,ct.c_void_p,ct.c_size_t],
        'cuMemcpyDtoH_v2':[ct.c_void_p,ct.c_uint64,ct.c_size_t],
        'cuLaunchKernel':[ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p],
    }
    for name,types in signatures.items():
        fn=getattr(driver,name);fn.argtypes=types;fn.restype=ct.c_int
    def call(name,*args):
        rc=getattr(driver,name)(*args)
        assert rc==0,(name,rc)
    call('cuInit',0)
    device=ct.c_int();previous=ct.c_void_p();context=ct.c_void_p()
    call('cuDeviceGet',ct.byref(device),0);call('cuCtxGetCurrent',ct.byref(previous))
    call('cuDevicePrimaryCtxRetain',ct.byref(context),device)
    call('cuCtxSetCurrent',context)
    module=ct.c_void_p();allocations=[]
    raw_outputs=[np.full(source.size+8,0xA55A,np.uint16) for _ in range(2)]
    try:
        ptx=ct.create_string_buffer(record['ptx'].encode())
        call('cuModuleLoadData',ct.byref(module),ct.cast(ptx,ct.c_void_p))
        function=ct.c_void_p()
        call('cuModuleGetFunction',ct.byref(function),module,record['entry'].encode())
        for host in (source,seed,*raw_outputs):
            pointer=ct.c_uint64();call('cuMemAlloc_v2',ct.byref(pointer),host.nbytes)
            allocations.append(pointer)
            call('cuMemcpyHtoD_v2',pointer,ct.c_void_p(host.ctypes.data),host.nbytes)
        values=[allocations[0],allocations[1],ct.c_uint64(allocations[2].value+8),
                ct.c_uint64(allocations[3].value+8),ct.c_int64(source.shape[0]),ct.c_int64(source.shape[1])]
        params=(ct.c_void_p*6)(*[ct.cast(ct.byref(value),ct.c_void_p) for value in values])
        call('cuLaunchKernel',function,1,1,1,128,1,1,0,None,params,None)
        call('cuCtxSynchronize')
        for pointer,host in zip(allocations[2:],raw_outputs,strict=True):
            call('cuMemcpyDtoH_v2',ct.c_void_p(host.ctypes.data),pointer,host.nbytes)
            np.testing.assert_array_equal(host[:4],np.uint16(0xA55A))
            np.testing.assert_array_equal(host[-4:],np.uint16(0xA55A))
        return tuple(host[4:-4].view(source.dtype).reshape(source.shape).copy() for host in raw_outputs)
    finally:
        if module.value:call('cuModuleUnload',module)
        for pointer in allocations:call('cuMemFree_v2',pointer)
        call('cuCtxSetCurrent',previous)
        call('cuDevicePrimaryCtxRelease_v2',device)


@pytest.mark.parametrize('storage',['f16','bf16'])
@pytest.mark.parametrize('profile',['zero','nan'])
def test_producer_preserves_declared_storage_relu_semantics(storage,profile,owning_gpu):
    import ml_dtypes
    dtype=np.float16 if storage=='f16' else ml_dtypes.bfloat16
    a=np.ones((2,32),dtype);b=np.ones((32,1),dtype);da=np.ones_like(a)
    fn=ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('a',))(product)
    try:
        fn.native_jvp(a,b,tangents=(da,))
        owner=next(iter(fn._native_prepared_jvp_calls.values()))
        source=np.zeros_like(a)
        source.view(np.uint16)[:,::2]=0x8000  # Negative zero in original storage.
        if profile=='nan':source.view(np.uint16)[1,0]=0x7E35 if storage=='f16' else 0x7FD3
        primal,tangent=_read_producer_storage(owner._members[0],source,da)
        expected=source.view(np.uint16)[0] if storage=='bf16' else np.zeros(32,np.uint16)
        np.testing.assert_array_equal(primal.view(np.uint16)[0],expected)
        if profile=='nan':assert np.isnan(primal[1]).all()
        else:np.testing.assert_array_equal(primal.view(np.uint16)[1],expected)
        np.testing.assert_array_equal(tangent.view(np.uint16),np.zeros_like(source,dtype=np.uint16))
    finally:fn.close_native_storage()


@pytest.mark.parametrize('root_index',[0,1,2])
@pytest.mark.parametrize('captured',[False,True])
def test_profile_refreshes_in_place_mutated_roots(package,root_index,captured):
    fn,original,_,_,_,_,permuted=package
    roots=tuple(x.copy() for x in original)
    fn.native_jvp(*roots[:2],tangents=(roots[2],))
    owner=next(iter(fn._native_prepared_jvp_calls.values()))
    stats=owner.scratch_stats()
    pointer=roots[root_index].ctypes.data
    roots[root_index][...] = roots[root_index]*-0.5+0.25
    assert roots[root_index].ctypes.data==pointer
    a,b=(roots[1],roots[0]) if permuted else roots[:2]
    expected,budgets=oracle(a,b,roots[2])
    outputs,timing=owner.profile(roots,repeats=3,captured=captured)
    check(outputs,expected,budgets)
    assert timing['program_ms']>=0 and all(x>=0 for x in timing['stage_ms'])
    assert owner.scratch_stats()==stats
