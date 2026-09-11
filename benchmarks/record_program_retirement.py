#!/usr/bin/env python3
"""Owning-device slot discovery and asynchronous SSD program retirement proof."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'python')]
import numpy as np  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from tessera.compiler.resident_object_pool import ResidentObjectPool  # noqa: E402
from tessera.compiler.resident_ssd import ResidentSSDProgram  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.control import vjp  # noqa: E402


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    device=Device(args.backend)
    P=ct.c_void_p
    create=getattr(device.lib,'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
    destroy=getattr(device.lib,'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
    create.argtypes,create.restype=[ct.POINTER(P),ct.c_uint],ct.c_int
    destroy.argtypes,destroy.restype=[P],ct.c_int
    stream=P()
    device.check(create(ct.byref(stream),1))
    options=dict(compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,
                 chip='sm_120' if device.cuda else 'gfx1151')
    memory=Memory(device)
    class Node:
        __slots__=('child',)
    node=Node()
    node.child=node
    try:
        with ResidentObjectPool.from_objects(node,stream=stream.value,allow_slots=True,**options) as pool:
            pool.collect(stream.value).wait()
            with pool.read(stream.value) as buffers:
                np.testing.assert_array_equal(memory.get(buffers[0])[:,2],[1])
        logical=lower_scheduled_ssd(3,1,2,2,2,compiler=args.compiler)
        with ResidentSSDProgram(logical,**options) as program:
            inputs=[memory.put(np.full(shape,.2,np.float32)) for shape in ((3,1,2),(3,1),(3,1,2),(3,1,2),(1,2,2))]
            seed=memory.put(np.ones((3,1,2),np.float32))
            program._forward._bound=program.forward.package.bind()
            program._reverse._bound=program.reverse.package.bind()
            def forbidden():
                raise AssertionError('healthy program retirement synchronized the context')
            for binding in (program._forward,program._reverse):
                binding._bound._sync=forbidden
            for _ in range(2):
                _,pullback=vjp(program,*inputs,stream=stream.value)
                gradients=pullback(seed)
                with gradients.read(stream.value):
                    pass
            identities=[program.forward.package.binding_digest,program.reverse.package.binding_digest]
            program.retire_async(stream.value)
            deadline=time.monotonic()+30
            polls=0
            while not program.poll_close():
                polls+=1
                if time.monotonic()>deadline:
                    raise TimeoutError('off-thread module retirement did not finish')
                time.sleep(.001)
            assert not program.frames and program._forward._bound is None and program._reverse._bound is None
        args.output.write_text(json.dumps(dict(backend=args.backend,artifacts=identities,
            slotted_cycle_verified=True,frames=2,public_api='vjp',off_thread_module_retirement=True,
            context_sync_forbidden=True,polls=polls,promotion_eligible=False,
            compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    finally:
        memory.close()
        device.check(destroy(stream))


if __name__=='__main__':main()
