#!/usr/bin/env python3
"""Owning-device numerical proof for the replay-bound serial SSD baseline."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    rows = []
    for chunk in (1,2,5):
        logical = lower_scheduled_ssd(5,2,3,2,chunk,compiler=args.compiler)
        program = materialize_ssd(logical,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),
                                  backend=args.backend,chip='sm_120' if device.cuda else 'gfx1151')
        rng = np.random.default_rng(740+chunk)
        inputs = [rng.uniform(-.5,.5,shape).astype(np.float32)
                  for shape in [(5,2,2),(5,2),(5,2,3),(5,2,3),(2,3,2)]]
        x,decay,b,c,state = inputs
        state = state.copy()
        y = np.empty_like(x)
        saved = []
        for t in range(5):
            state = decay[t,:,None,None]*state+b[t,:,:,None]*x[t,:,None,:]
            y[t] = (c[t,:,:,None]*state).sum(axis=1)
            if (t+1)%chunk == 0 or t == 4:
                saved.append(state.copy())
        expected = [y,state,np.array(saved)]
        outputs = [np.full_like(v,np.nan) for v in expected]
        pointers,views = [],[]
        binding = program.bind()
        try:
            for value in inputs+outputs:
                pointer = ct.c_void_p()
                device.check(device.alloc(ct.byref(pointer),value.nbytes)); pointers.append(pointer)
                device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda
                             else device.copy(pointer,value.ctypes.data,value.nbytes,1))
                views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,
                             typestr=value.dtype.str,data=(pointer.value,False))))
            binding(*views,1)
            observed = []
            for i,value in enumerate(inputs+outputs):
                result = np.empty_like(value)
                device.check(device.dtoh(result.ctypes.data,pointers[i],result.nbytes) if device.cuda
                             else device.copy(result.ctypes.data,pointers[i],result.nbytes,2))
                if i < 5:
                    np.testing.assert_array_equal(result,value)
                else:
                    np.testing.assert_allclose(result,expected[i-5],rtol=1e-5,atol=1e-6)
                    observed.append(float(np.max(np.abs(result-expected[i-5]))))
            rows.append(dict(chunk=chunk,max_abs_errors=observed,binding_digest=program.package.binding_digest,
                             image_sha256=hashlib.sha256(program.package.image).hexdigest()))
        finally:
            binding.close()
            for pointer in reversed(pointers):
                device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(schema=1,backend=args.backend,
        architecture='sm_120' if device.cuda else 'gfx1151',
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        execution='native_gpu',rows=rows,promotion_eligible=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
