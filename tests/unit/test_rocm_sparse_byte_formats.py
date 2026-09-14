"""Separate exact-device gates for byte-sized sparse storage."""
import os
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_sparse_runtime import compile_sparse_matmul


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1', reason='owning gfx1201 proof')
@pytest.mark.parametrize('dtype,rhs_dtype,mnemonic', [
    (np.int8,np.int8,'i32_16x16x32_iu8'),
    (np.uint8,np.uint8,'i32_16x16x32_iu8'),
    (np.uint8,np.int8,'i32_16x16x32_iu8'),
    (np.int8,np.uint8,'i32_16x16x32_iu8'),
    (ml_dtypes.float8_e4m3fn,ml_dtypes.float8_e4m3fn,'f32_16x16x32_fp8_fp8'),
    (ml_dtypes.float8_e5m2,ml_dtypes.float8_e5m2,'f32_16x16x32_bf8_bf8'),
    (ml_dtypes.float8_e4m3fn,ml_dtypes.float8_e5m2,'f32_16x16x32_fp8_bf8'),
    (ml_dtypes.float8_e5m2,ml_dtypes.float8_e4m3fn,'f32_16x16x32_bf8_fp8'),
])
@pytest.mark.parametrize('shape', [(16,16,32),(32,48,64)])
def test_logical_byte_sparse_device(dtype,rhs_dtype,mnemonic,shape,tmp_path):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == 'gfx1201'
    m,n,k = shape
    package = compile_sparse_matmul(m,n,k,dtype=np.dtype(dtype).name,rhs_dtype=np.dtype(rhs_dtype).name,
                                   accum='i32' if dtype in (np.int8,np.uint8) else 'f32')
    binary = tmp_path/'sparse.hsaco'
    binary.write_bytes(package.image)
    assembly = subprocess.check_output([str(Path(os.environ['TESSERA_LLVM_BIN'])/'llvm-objdump'),'-d',str(binary)],text=True)
    assert 'v_swmmac_' + mnemonic in assembly.lower()
    rng = np.random.default_rng(254)
    a = (rng.integers(128,256,size=(m,k)) if dtype == np.uint8 else rng.integers(-4,5,size=(m,k))).astype(dtype)
    # Rotate the selected pair; exercise every sparse index pair across rows.
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for row in range(m):
        for group in range(k//4):
            keep = pairs[(row+group)%len(pairs)]
            for col in range(4):
                if col not in keep:
                    a[row,group*4+col] = 0
    b = (rng.integers(128,256,size=(k,n)) if rhs_dtype == np.uint8 else rng.integers(-4,5,size=(k,n))).astype(rhs_dtype)
    oracle_type = np.int32 if dtype in (np.int8,np.uint8) else np.float32
    np.testing.assert_array_equal(package.run(a,b),a.astype(oracle_type)@b.astype(oracle_type))
    a[-1,-4:] = 1
    with pytest.raises(ValueError,match='invalid 2:4'):
        package.run(a,b)


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1', reason='owning gfx1201 proof')
@pytest.mark.parametrize('dtype,rhs_dtype',[(np.int8,np.int8),(np.uint8,np.uint8),(np.int8,np.uint8),(np.uint8,np.int8)])
def test_int4_logical_packing_device(dtype,rhs_dtype,tmp_path,monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == 'gfx1201'
    package = compile_sparse_matmul(32,48,64,dtype=np.dtype(dtype).name,
        rhs_dtype=np.dtype(rhs_dtype).name,accum='i32',integer_bits=4)
    binary = tmp_path/'int4.hsaco'
    binary.write_bytes(package.image)
    asm = subprocess.check_output([str(Path(os.environ['TESSERA_LLVM_BIN'])/'llvm-objdump'),'-d',str(binary)],text=True)
    assert 'v_swmmac_i32_16x16x32_iu4' in asm.lower()
    rng = np.random.default_rng(104)
    a = rng.integers(-8 if dtype == np.int8 else 0,8 if dtype == np.int8 else 16,size=(32,64)).astype(dtype)
    b = rng.integers(-8 if rhs_dtype == np.int8 else 0,8 if rhs_dtype == np.int8 else 16,size=(64,48)).astype(rhs_dtype)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for r in range(32):
        for g in range(16):
            for j in range(4):
                if j not in pairs[(r+g)%6]: a[r,4*g+j] = 0
    np.testing.assert_array_equal(package.run(a,b),a.astype(np.int32)@b.astype(np.int32))
    b[0,0] = 8 if rhs_dtype == np.int8 else 16
    with pytest.raises(ValueError,match='declared range'): package.run(a,b)
    # Bypass only the host preflight to prove the compiled validity guard.
    monkeypatch.setattr(type(package),'_inputs',lambda self,a,b: (a,b))
    with pytest.raises(ValueError,match='invalid 2:4'): package.run(a,b)
