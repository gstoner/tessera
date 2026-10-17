"""Owning gfx1201 execution of the complete native seven-member MLP."""
import json
import ml_dtypes
import numpy as np
import pytest
from benchmarks.rocm.benchmark_public_quantized_gate_up import inputs as gate_inputs,oracle as gate_oracle
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import reference_nvfp4_expand,nvfp4_expansion_policy
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
from tests.unit.test_native_quantized_mlp_program import source

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_rocm,pytest.mark.compiler_route]


def inputs(m,h,k,n,seed=731):
    roots=gate_inputs(m,h,k,seed)
    rng=np.random.default_rng(seed+1)
    return roots+[rng.integers(0,256,size=(n,h//2),dtype=np.uint8),
        rng.uniform(.125,2,size=(n,h//16)).astype(ml_dtypes.float8_e4m3fn),
        np.array([.0123456789],np.float64)]


def down_reference(roots,activation):
    m,h=activation.shape;n=roots[4].shape[0]
    blocks=activation.astype(np.float32).reshape(m,h//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(*roots[4:],row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
    quantized=q*sa.astype(np.float64)[...,None]
    weights=b.astype(np.float64).reshape(n,h//16,16)*sb.astype(np.float64).T[...,None]
    return np.einsum('mgj,ngj->mn',quantized,weights),quantized,weights


def oracle(roots):
    output,_,_=down_reference(roots,gate_oracle(roots[:4]))
    return output.astype(np.float32).astype(ml_dtypes.bfloat16)


def assert_compiled_chain(package,roots,actual,lib):
    # Diagnostic staged owners prove resident wiring independently. Production
    # execution above is one seven-step owner and has no intermediate readback.
    with package.gate.prepare(roots[:4],runtime_library=lib) as owner:
        generation,_=owner.invoke();activation=owner.read(generation)[0]
    expected_activation=gate_oracle(roots[:4])
    # Validate at the explicit parent store: a single BF16 midpoint crossing
    # can be amplified by SiLU even when its implementation is exact.
    x,packed,weight_scales,globals_=roots[:4];m,k=x.shape;h=activation.shape[1]
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).reshape(m,k)
    b,sb=reference_nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,h,2*h],numeric_policy=nvfp4_expansion_policy())
    parent_reference=np.einsum('mgj,ngj,mg,gn->mn',q.astype(np.float64).reshape(m,k//16,16),
        b.astype(np.float64).reshape(2*h,k//16,16),sa.astype(np.float64),sb.astype(np.float64))
    from tests.device.rocm.test_fp8_blockscale_w8a8 import _launch
    parent=_launch(package.gate.product,q,b.T,sa,sb,package.gate.product_recipe.shape).astype(np.float32)
    parent_stored=parent_reference.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(parent,parent_stored,rtol=.015625,atol=.00002)
    gate,up=parent[:,:h],parent[:,h:]
    activation_at_parent=(gate/(np.float32(1)+np.exp(-gate))*up).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(activation.astype(np.float32),activation_at_parent.astype(np.float32),rtol=.015625,atol=.00002)
    with package.down.prepare([activation,*roots[4:]],runtime_library=lib) as owner:
        generation,_=owner.invoke();staged=owner.read(generation)[0]
    np.testing.assert_array_equal(actual,staged)
    live,q_live,weights=down_reference(roots,activation)
    reference,q_reference,_=down_reference(roots,expected_activation)
    live_stored=live.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    reference_stored=reference.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    # The existing per-stage gate is unchanged. Quantization is discontinuous:
    # carrying a tolerated BF16 perturbation through a second quantizer cannot
    # reuse the same pointwise relative gate against the ideal earlier value.
    np.testing.assert_allclose(actual.astype(np.float32),live_stored,rtol=.015625,atol=.00002)
    propagation=np.einsum('mgj,ngj->mn',np.abs(q_live-q_reference),np.abs(weights))
    store_rounding=np.abs(live_stored-live)+np.abs(reference_stored-reference)
    numerical=.015625*np.abs(live_stored)+.00002
    assert np.all(np.abs(actual.astype(np.float64)-reference_stored)<=propagation+store_rounding+numerical)


def provider():
    assert rt._rocm_live_arch()=='gfx1201'
    lib=rt._load_rocm_native_movement_runtime()
    assert lib is not None
    return lib._name


@pytest.mark.parametrize('shape',[(1,16,16,1),(3,32,48,19),(256,128,12288,256)])
def test_full_native_mlp_correctness_portable_replay_and_root_updates(shape,monkeypatch):
    lib=provider();m,h,k,n=shape
    package=package_quantized_product_graph(source(m,h,k,n))
    roots=inputs(*shape)
    before=[x.copy() for x in roots]
    with package.prepare(roots,runtime_library=lib) as owner:
        generation,_=owner.invoke()
        held=owner.read(generation)[0]
        assert_compiled_chain(package,roots,held,lib)
        for a,b in zip(roots,before,strict=True):np.testing.assert_array_equal(a,b)
    held_before=held.copy()
    serialized=json.loads(json.dumps(package.to_manifest()))
    monkeypatch.setenv('TESSERA_OPT','/missing/compiler')
    monkeypatch.setenv('PATH','/missing/tools')
    restored=NativeQuantizedMLPProgram.from_manifest(serialized)
    changed=inputs(*shape,seed=901)
    with restored.prepare(changed,runtime_library=lib) as owner:
        with pytest.raises(RuntimeError,match='10'):
            owner.invoke_range(6,1)
        generation,_=owner.invoke()
        actual=owner.read(generation)[0]
        assert_compiled_chain(restored,changed,actual,lib)
        actual_before=actual.copy()
        updated=inputs(*shape,seed=1123)
        owner.update(updated)
        with pytest.raises(RuntimeError,match='10'):
            owner.invoke_range(6,1)
        generation,_=owner.invoke()
        refreshed=owner.read(generation)[0]
        assert_compiled_chain(restored,updated,refreshed,lib)
        np.testing.assert_array_equal(actual,actual_before)
    np.testing.assert_array_equal(held,held_before)


@pytest.mark.parametrize('shape',[(1,16,16,1),(3,32,48,19),(256,128,12288,256)])
def test_public_mlp_has_seven_native_receipts_and_no_eager_math(shape,monkeypatch):
    import tessera as ts
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp
    lib=provider();m,h,k,n=shape;roots=inputs(*shape)
    fn=ts.jit(target='rocm_gfx1201',native_required=True)(make_mlp(m,h,k,n))
    from tessera.compiler import nvfp4_expansion
    with monkeypatch.context() as patch:
        def forbidden(*a,**kw):raise AssertionError('public MLP executed eager numerical work')
        for name in ('max','maximum','exp'):patch.setattr(np,name,forbidden)
        patch.setattr(nvfp4_expansion,'reference_nvfp4_expand',forbidden)
        first=fn(*roots)
    assert fn.execution_kind=='native_gpu'
    assert len(fn._rocm_quantized_product_last_receipts)==7
    program=fn._rocm_quantized_product_last_program.native
    assert_compiled_chain(program,roots,first,lib)
    report=fn.compile_report()
    assert '7 ordered native packages' in report.target_decision['rocm_gfx1201']
    assert all(len(report.ir_hashes[key])==64 for key in ('graph_ir','schedule_ir','tile_ir','target_ir'))
    artifact=fn.runtime_artifact()
    assert artifact.metadata['compiler_path']=='canonical_rocm_quantized_mlp_program'
    monkeypatch.setenv('TESSERA_OPT','/missing/compiler')
    changed=inputs(*shape,seed=1123)
    second=fn(*changed)
    assert_compiled_chain(program,changed,second,lib)
    replay=rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()),changed)
    assert replay.get('ok') and replay.get('execution_kind')=='native_gpu',replay
    assert len(replay['component_receipts'])==7
    np.testing.assert_array_equal(replay['output'],second)


def test_public_mlp_fresh_process_replay_without_compiler(tmp_path):
    import os
    import subprocess
    import sys
    import tessera as ts
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp
    provider();roots=inputs(3,32,48,19)
    fn=ts.jit(target='rocm_gfx1201',native_required=True)(make_mlp())
    expected=fn(*roots)
    (tmp_path/'artifact.json').write_text(fn.runtime_artifact().to_json())
    np.savez(tmp_path/'roots.npz',x=roots[0].astype(np.float32),packed=roots[1],scales=roots[2].view(np.uint8),globals_=roots[3],
        down_packed=roots[4],down_scales=roots[5].view(np.uint8),down_globals=roots[6])
    script='\n'.join([
        'import sys,numpy as np,ml_dtypes',
        'from pathlib import Path',
        'from tessera import runtime as rt',
        "root=Path(sys.argv[1]);p=np.load(root/'roots.npz')",
        "artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())",
        "roots=[p['x'].astype(ml_dtypes.bfloat16),p['packed'],p['scales'].view(ml_dtypes.float8_e4m3fn),p['globals_'],p['down_packed'],p['down_scales'].view(ml_dtypes.float8_e4m3fn),p['down_globals']]",
        'result=rt.launch(artifact,roots)',
        "assert result.get('ok') and result.get('execution_kind')=='native_gpu',result",
        "assert len(result['component_receipts'])==7",
        "np.save(root/'result.npy',result['output'])",
    ])
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,
        env={**os.environ,'TESSERA_OPT':'/missing/compiler','PATH':'/missing/tools'})
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/'result.npy').view(np.uint16),expected.view(np.uint16))


def test_nested_mlp_inspection_metadata_cannot_poison_warm_program(monkeypatch):
    import tessera as ts
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp
    lib=provider();roots=inputs(3,32,48,19)
    fn=ts.jit(target='rocm_gfx1201',native_required=True)(make_mlp())
    fn(*roots);artifact=fn.runtime_artifact()
    artifact.metadata['native_program']['native_program']['gate']['stages'][3]['descriptor']['provenance']['native_parent_view']['member_json']='{}'
    result=rt.launch(artifact,roots)
    assert not result.get('ok'),result
    monkeypatch.setenv('TESSERA_OPT','/missing/compiler')
    changed=inputs(3,32,48,19,seed=119)
    actual=fn(*changed)
    assert_compiled_chain(fn._rocm_quantized_product_last_program.native,changed,actual,lib)
