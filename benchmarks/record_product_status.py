#!/usr/bin/env python3
"""Exact-device checked CFG products and native scoped-reader composition."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_dynamic_cfg_storage import switch_source  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def function_cfg_source():
    from benchmarks.record_dynamic_cfg_storage import switch_source
    source=switch_source()
    source=source.replace('attributes {tessera.autodiff = "reverse"}',
        'attributes {tessera.autodiff = "reverse", tessera.structured_cfg.max_steps = 2 : i64, '
        'tessera.structured_cfg.digest = "'+'a'*64+'"}')
    source=source.replace('%y = scf.execute_region -> tensor<4xf32> {','')
    source=source.replace('scf.yield','return')
    suffix='''} {tessera.structured_cfg.max_steps = 2 : i64,
           tessera.structured_cfg.digest = "'''+ 'a'*64 +'''"}
        return %y : tensor<4xf32>'''
    return source.replace(suffix,'')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    d=Device(args.backend); chip='sm_120' if d.cuda else 'gfx1151'
    def build(source,checked=False):
        return materialize_persistent_tape(source,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,checked_status=checked)
    pointers=[]
    def upload(a):
        p=ct.c_void_p();d.check(d.alloc(ct.byref(p),a.nbytes));pointers.append(p)
        d.check(d.htod(p,a.ctypes.data,a.nbytes) if d.cuda else d.copy(p,a.ctypes.data,a.nbytes,1))
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=a.shape,typestr=a.dtype.str,data=(p.value,False)))
    def download(v):
        a=np.empty(4,np.float32);p=v.__cuda_array_interface__['data'][0]
        d.check(d.dtoh(a.ctypes.data,p,a.nbytes) if d.cuda else d.copy(a.ctypes.data,p,a.nbytes,2))
        return a
    rows=[]; streams=[]
    try:
        pair=build(switch_source(),True)
        for x0 in (.5,1.5,2.5):
            x=np.full(4,x0,np.float32);seed=upload(np.ones(4,np.float32))
            with pair.capture(upload(x)) as frame:
                primal=x*x if x0<1 else (2*x if x0<2 else x*x*x)
                derivative=2*x if x0<1 else (np.full_like(x,2) if x0<2 else 3*x*x)
                np.testing.assert_allclose(download(frame.primals[0]),primal)
                for _ in range(2):
                    np.testing.assert_allclose(download(frame.backward(seed)[0]),derivative)
            rows.append(dict(case='switch',input=x0,repeated_backward=2))
        imported=build(function_cfg_source(),True)
        with imported.capture(upload(np.full(4,2.5,np.float32))) as frame:
            np.testing.assert_allclose(download(frame.primals[0]),np.full(4,15.625,np.float32))
            np.testing.assert_allclose(download(frame.backward(upload(np.ones(4,np.float32)))[0]),np.full(4,18.75,np.float32))
        rows.append(dict(case='imported_function_cfg',forward=15.625,reverse=18.75))
        failed=build(switch_source().replace('max_steps = 2','max_steps = 1'),True)
        try:failed.capture(upload(np.ones(4,np.float32)))
        except RuntimeError as error:
            assert 'guard failed' in str(error)
        else:raise AssertionError('exhaustion exposed output')
        rows.append(dict(case='exhaustion',output_refused=True))
        square='''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          return %y : tensor<4xf32> } }'''
        pair=build(square)
        create=getattr(d.lib,'cuStreamCreate' if d.cuda else 'hipStreamCreateWithFlags')
        create.argtypes,create.restype=[ct.POINTER(ct.c_void_p),ct.c_uint],ct.c_int
        destroy=getattr(d.lib,'cuStreamDestroy_v2' if d.cuda else 'hipStreamDestroy')
        destroy.argtypes,destroy.restype=[ct.c_void_p],ct.c_int
        for _ in range(3):
            s=ct.c_void_p();d.check(create(ct.byref(s),1));streams.append(s)
        with pair.capture(upload(np.full(4,2,np.float32))) as first, pair.capture(upload(np.full(4,3,np.float32))) as second:
            seed=upload(np.ones(4,np.float32))
            generation=first.backward_async(streams[0].value,seed,tracked=True)
            child=generation.backward_into(second,streams[1].value)
            generation.retire(streams[2].value).wait()
            targets=[upload(np.zeros(v.shape,dtype=v.typestr)) for v in first._outputs]
            consumed=child.submit_to(first._bindings[0],streams[1].value,*targets,1)
            child.retire(streams[2].value).wait()
            consumed.wait()
            np.testing.assert_allclose(download(targets[0]),np.full(4,576,np.float32))
        rows.append(dict(case='scoped_reverse_composition',derivative=24,streams=3))
    finally:
        for p in pointers:d.check(d.free(p))
        for s in streams:d.check(destroy(s))
    names=['src/transforms/lib/AutodiffPairedPass.cpp','src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_persistent_tape.py',
           'python/tessera/compiler/native_reader_retirement.py','python/tessera/compiler/native_gpu_tensor.py']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,rows=rows,promotion_eligible=False,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Checked switch reverse, exhaustion refusal and scoped reverse composition passed')


if __name__=='__main__':main()
