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


def oracle(roots):
    activation=gate_oracle(roots[:4]);m,h=activation.shape;n=roots[4].shape[0]
    blocks=activation.astype(np.float32).reshape(m,h//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(*roots[4:],row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
    output=np.einsum('mgj,ngj,mg,gn->mn',q,b.astype(np.float64).reshape(n,h//16,16),sa.astype(np.float64),sb.astype(np.float64))
    return output.astype(np.float32).astype(ml_dtypes.bfloat16)


def provider():
    assert rt._rocm_live_arch()=='gfx1201'
    lib=rt._load_rocm_native_movement_runtime()
    assert lib is not None
    return lib._name


@pytest.mark.parametrize('shape',[(1,16,16,1),(3,32,48,19),(256,128,12288,256)])
def test_full_native_mlp_correctness_portable_replay_and_root_updates(shape,monkeypatch):
    lib=provider();m,h,k,n=shape
    package=package_quantized_product_graph(source(m,h,k,n))
    roots=inputs(*shape);expected=oracle(roots)
    before=[x.copy() for x in roots]
    with package.prepare(roots,runtime_library=lib) as owner:
        generation,_=owner.invoke()
        held=owner.read(generation)[0]
        np.testing.assert_allclose(held.astype(np.float32),expected.astype(np.float32),rtol=.015625,atol=.00002)
        for a,b in zip(roots,before,strict=True):np.testing.assert_array_equal(a,b)
    held_before=held.copy()
    serialized=json.loads(json.dumps(package.to_manifest()))
    monkeypatch.setenv('TESSERA_OPT','/missing/compiler')
    monkeypatch.setenv('PATH','/missing/tools')
    restored=NativeQuantizedMLPProgram.from_manifest(serialized)
    changed=inputs(*shape,seed=901);expected2=oracle(changed)
    with restored.prepare(changed,runtime_library=lib) as owner:
        with pytest.raises(RuntimeError,match='10'):
            owner.invoke_range(6,1)
        generation,_=owner.invoke()
        actual=owner.read(generation)[0]
        np.testing.assert_allclose(actual.astype(np.float32),expected2.astype(np.float32),rtol=.015625,atol=.00002)
    np.testing.assert_array_equal(held,held_before)
