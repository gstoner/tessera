"""Logical packing and multi-tile SWMMAC proof; no performance admission."""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_sparse_logical import sparse_logical_schedule_ir
from tests._support import rocm_isa
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize('shape', [(0,16,32), (16,17,32), (16,16,31), (True,16,32)])
def test_sparse_logical_refuses_unsupported_envelope(shape):
    with pytest.raises(ValueError, match='multiples'):
        sparse_logical_schedule_ir(*shape, 'float16')


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_sparse_logical_native_ancestry(dtype):
    from tests._support.compiler_tool import require_tessera_opt
    compiler = require_tessera_opt('tessera-schedule-to-tile', 'lower-tile-to-rocm',
                                   'lower-tessera-target-to-rocdl')
    source = sparse_logical_schedule_ir(32,48,64,dtype)
    tile = subprocess.check_output([compiler,'--tessera-schedule-to-tile'],input=source,text=True)
    assert 'tile.sparse_mma' in tile and 'schedule.sparse_mma' not in tile
    target = subprocess.check_output([compiler,'--lower-tile-to-rocm'],input=tile,text=True)
    assert 'tessera_rocm.swmmac' in target
    llvm = subprocess.check_output([compiler,'--lower-tessera-target-to-rocdl'],input=target,text=True)
    assert 'llvm.amdgcn.swmmac' in llvm


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1', reason='owning gfx1201 host opt-in')
@pytest.mark.parametrize('shape', [(16,16,32),(32,48,64),(48,32,128)])
@pytest.mark.parametrize('dtype', [np.float16,ml_dtypes.bfloat16])
def test_sparse_logical_device_packing_and_k_accumulation(shape,dtype,tmp_path):
    from tessera import runtime as rt
    from tessera.compiler.native_gpu_storage import _decode_image
    assert rt._rocm_live_arch() == 'gfx1201'
    m,n,k = shape
    source = sparse_logical_schedule_ir(m,n,k,np.dtype(dtype).name)
    env = {key: val for key,val in os.environ.items() if not key.startswith('ROCP') and key != 'LD_PRELOAD'}
    lowered = subprocess.check_output([os.environ['TESSERA_OPT'],'--tessera-schedule-to-tile',
        '--lower-tile-to-rocm','--lower-tessera-target-to-rocdl'],input=source,text=True,env=env)
    llvm = Path(os.environ['TESSERA_LLVM_BIN'])
    pipeline = 'builtin.module(gpu.module(convert-vector-to-llvm,convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary{toolkit='+os.environ['ROCM_PATH']+'})'
    binary = subprocess.check_output([str(llvm/'mlir-opt'),'--pass-pipeline='+pipeline],input=lowered,text=True,env=env)
    image = _decode_image(re.findall(r'"((?:\\.|[^"\\])*)"',binary)[-1])
    storage = 'f16' if dtype == np.float16 else 'bf16'
    rocm_isa.assert_selected(image, chip='gfx1201', pattern=r'v_swmmac_\w+',
        require='v_swmmac_f32_16x16x32_' + storage,
        forbid='v_swmmac_f32_16x16x32_' + ('bf16' if storage == 'f16' else 'f16'),
        what=f'logical sparse {np.dtype(dtype).name}')
    hip = rt._load_hip_for_launch()
    P = ct.c_void_p
    def check(status):
        assert status == 0, f'HIP status {status}'
    module,fn = P(),P()
    blob = ct.create_string_buffer(image)
    check(hip.hipModuleLoadData(ct.byref(module),blob))
    check(hip.hipModuleGetFunction(ct.byref(fn),module,b'probe'))
    rng = np.random.default_rng(7)
    a = np.zeros((m,k),dtype)
    pairs = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    for row in range(m):
        for group in range(k//4):
            pair = pairs[(row+group)%6]
            a[row,group*4+np.array(pair)] = (rng.integers(-4,5,size=2)/4).astype(dtype)
    b = (rng.integers(-4,5,size=(k,n))/4).astype(dtype)
    output = np.zeros((m,n),np.float32)
    status = np.zeros((m//16)*(n//16)*32,np.int32)
    hosts = (a,b,output,status)
    ptrs = []
    try:
        for host in hosts:
            ptr = P()
            check(hip.hipMalloc(ct.byref(ptr),host.nbytes))
            ptrs.append(ptr)
        def run():
            for ptr,host in zip(ptrs,hosts,strict=True):
                check(hip.hipMemcpy(ptr,host.ctypes.data_as(P),host.nbytes,1))
            values = []
            for ptr,host in zip(ptrs,hosts,strict=True):
                values.extend((P(ptr.value),P(ptr.value),ct.c_int64(0),ct.c_int64(host.size),ct.c_int64(1)))
            args = (P*len(values))(*[ct.cast(ct.byref(v),P) for v in values])
            check(hip.hipModuleLaunchKernel(fn,(m//16)*(n//16),1,1,32,1,1,0,None,args,None))
            check(hip.hipDeviceSynchronize())
            for ptr,host in zip(ptrs[2:],hosts[2:],strict=True):
                check(hip.hipMemcpy(host.ctypes.data_as(P),ptr,host.nbytes,2))
        run()
        np.testing.assert_array_equal(status,1)
        np.testing.assert_array_equal(output,a.astype(np.float32)@b.astype(np.float32))
        # An invalid group in the last K tile cannot be silently pruned. Each
        # output-column tile reports it, even after valid earlier iterations.
        a[-1,-4:] = 1
        run()
        assert np.count_nonzero(status == 0) == n//16
        if destination := os.environ.get('TESSERA_SPARSE_PROOF_DIR'):
            root = Path(destination)
            root.mkdir(parents=True,exist_ok=True)
            record = dict(target='gfx1201', shape=list(shape), dtype=np.dtype(dtype).name,
                          source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          image_sha256=hashlib.sha256(image).hexdigest(),
                          compiler_sha256=hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
                          max_abs_error=0.0, invalid_group_rejected=True,
                          packing='logical_row_major_to_SWMMAC_in_compiled_IR',
                          production_admission=False, performance_eligible=False)
            (root/f'{np.dtype(dtype).name}_{m}_{n}_{k}.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:
        check(hip.hipDeviceSynchronize())
        for ptr in reversed(ptrs):
            check(hip.hipFree(ptr))
        check(hip.hipModuleUnload(module))
