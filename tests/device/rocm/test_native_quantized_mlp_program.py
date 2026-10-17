"""Owning gfx1201 execution of the complete native seven-member MLP."""
import json
import ml_dtypes
import numpy as np
import pytest
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
from tests.unit.test_native_quantized_mlp_program import source

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_rocm,pytest.mark.compiler_route]




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
