#!/usr/bin/env python3
"""Owning-device GPU heap transaction and SSD checkpoint VJP proof."""
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
from tessera.compiler.gpu_exception_heap import materialize_gpu_exception_heap  # noqa: E402
from tessera.compiler.source_exception_heap import pack_exception_table  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402


def execute(device,program,inputs,outputs,*indices):
    pointers,views = [],[]
    binding = program.bind()
    try:
        for value in inputs+outputs:
            pointer = ct.c_void_p()
            device.check(device.alloc(ct.byref(pointer),value.nbytes))
            pointers.append(pointer)
            device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda else device.copy(pointer,value.ctypes.data,value.nbytes,1))
            views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False))))
        binding(*views,*indices,1)
        observed = []
        for index,value in enumerate(inputs+outputs):
            got = np.empty_like(value)
            device.check(device.dtoh(got.ctypes.data,pointers[index],got.nbytes) if device.cuda else device.copy(got.ctypes.data,pointers[index],got.nbytes,2))
            if index < len(inputs):
                np.testing.assert_array_equal(got,value)
            else:
                observed.append(got)
        return observed
    finally:
        binding.close()
        for pointer in reversed(pointers):
            device.check(device.free(pointer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    options = dict(compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip='sm_120' if device.cuda else 'gfx1151')
    table = pack_exception_table([('ValueError',['@tensor','a']),('RuntimeError',['@tensor','b'])])
    contract = dict(exception_heap=table,error_dynamic=True,error_payload_sites=['a','b'])
    source = 'module attributes {tessera.source_state = '+json.dumps(json.dumps(contract))+'} {}'
    heap = materialize_gpu_exception_heap(source,capacity=8,**options)
    cases = []
    for generation,lengths in enumerate(([2,3],[-1,3],[8,1],[0,0],[(1<<63)-1,1],[3,1]),1):
        payload,records,status = execute(device,heap,[np.array(lengths,np.int64),np.arange(8,dtype=np.float32)],
            [np.full(8,-9,np.float32),np.full((2,3),-9,np.int64),np.full(2,-9,np.int64)],generation)
        valid = min(lengths)>=0 and sum(lengths)<=8
        assert status[0] == (0 if valid else 1)
        if valid:
            assert status[1] == sum(lengths)
            error = heap.decode_completed(1,status,records,payload,generation=generation)
            np.testing.assert_array_equal(error.args[0],np.arange(lengths[0],dtype=np.float32))
            try:
                heap.decode_completed(1,status,records,payload,generation=generation+1)
            except ValueError:
                pass
            else:
                raise AssertionError('stale generation accepted')
        else:
            try:
                heap.decode_completed(1,status,records,payload,generation=generation)
            except MemoryError:
                pass
            else:
                raise AssertionError('failed allocation exposed an exception payload')
            np.testing.assert_array_equal(payload,np.full(8,-9,np.float32))
            np.testing.assert_array_equal(records,np.full((2,3),-9,np.int64))
        cases.append(dict(lengths=lengths,status=status.tolist(),generation=generation))
    rng = np.random.default_rng(742)
    values = [rng.uniform(-.4,.4,s).astype(np.float32) for s in [(3,2,2),(3,2),(3,2,2),(3,2,2),(2,2,2)]]
    def forward(inputs):
        x,d,b,c,state = inputs
        state = state.copy()
        ys,cps = [],[]
        for t in range(3):
            state = d[t,:,None,None]*state+b[t,:,:,None]*x[t,:,None,:]
            ys.append((c[t,:,:,None]*state).sum(axis=1))
            if t in (1,2): cps.append(state.copy())
        return [np.array(ys),state,np.array(cps)]
    expected = forward(values)
    logical = lower_scheduled_ssd(3,2,2,2,2,compiler=args.compiler)
    fwd = materialize_ssd(logical,cooperative=True,**options)
    actual = execute(device,fwd,values,[np.empty_like(v) for v in expected])
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-6)
    seeds = [rng.uniform(-.4,.4,v.shape).astype(np.float32) for v in expected]
    bwd = materialize_ssd(logical,adjoint=True,**options)
    grads = execute(device,bwd,values+[actual[2]]+seeds,[np.empty_like(v) for v in values])
    errors = []
    def loss(inputs):
        return sum(np.sum(a*b) for a,b in zip(forward(inputs),seeds,strict=True))
    for i,value in enumerate(values):
        numerical = np.empty_like(value)
        for index in np.ndindex(value.shape):
            plus,minus = [v.astype(np.float64) for v in values],[v.astype(np.float64) for v in values]
            plus[i][index]+=1e-5
            minus[i][index]-=1e-5
            numerical[index]=(loss(plus)-loss(minus))/2e-5
        np.testing.assert_allclose(grads[i],numerical,rtol=2e-5,atol=2e-6)
        errors.append(float(np.max(np.abs(grads[i]-numerical))))
    args.output.write_text(json.dumps(dict(schema=1,backend=args.backend,chip=options['chip'],
        execution='native_gpu',compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        heap_binding=heap.package.binding_digest,forward_binding=fwd.package.binding_digest,backward_binding=bwd.package.binding_digest,
        heap_cases=cases,gradient_max_abs_errors=errors,promotion_eligible=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
