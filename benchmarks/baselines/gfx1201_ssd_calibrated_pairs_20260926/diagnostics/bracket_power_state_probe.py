"""Diagnostic (not evidence): attribute the gfx1201 marker-bracketing overhead.

Same clean SSD image, same process, four window protocols, HIP events only:
  A plain          : event, N launches, event
  B plain+gap      : H2D 16-byte copy + hipDeviceSynchronize, then A
  C markers        : event, marker, N launches, marker, event (span reset once)
  D recorder       : B's gap, then C (what record_ssd_gpu does)
  E plain+idle     : 2 ms host sleep, then A
"""
import ctypes as ct, statistics, sys, time, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path[:0] = ['.', 'python']
from benchmarks.record_device_ring_protocol import Device
from benchmarks.record_ssd_gpu import _llvm_bin, _rocm_identity
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd
from tessera.compiler.native_ssd import materialize_ssd
from tessera.compiler.native_device_clock import build_device_clock_marker

coop = sys.argv[1] == 'cooperative'
compiler = Path(os.environ['TESSERA_OPT'])
T, H, N, P, chunk = 32, 2, 16, 4, 8
device = Device('rocm')
chip = _rocm_identity(device)['architecture']
logical = lower_scheduled_ssd(T, H, N, P, chunk, compiler=compiler)
program = materialize_ssd(logical, compiler=compiler, llvm_bin=_llvm_bin(), backend='rocm', chip=chip, cooperative=coop)
rng = np.random.default_rng(748)
inputs = [rng.uniform(-.5, .5, s).astype(np.float32) for s in [(T,H,P),(T,H),(T,H,N),(T,H,N),(H,N,P)]]
outputs = [np.zeros((T,H,P),np.float32), np.zeros((H,N,P),np.float32), np.zeros((T//chunk,H,N,P),np.float32)]
binding = program.bind()
pointers, views = [], []
for v in inputs + outputs:
    p = ct.c_void_p(); device.check(device.alloc(ct.byref(p), v.nbytes)); pointers.append(p)
    device.check(device.copy(p, v.ctypes.data, v.nbytes, 1))
    views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=v.shape, typestr=v.dtype.str, data=(p.value, False))))
binding(*views, 1)
raw, _, grid, block, _ = binding._resident(*views, 1)
values, shared = binding._bound._launch_size(raw, grid, block)
argv = (ct.c_void_p*len(values))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in values])
marker = build_device_clock_marker(compiler=compiler, llvm_bin=_llvm_bin(), backend='rocm', chip=chip)
Pv = ct.c_void_p
module, fn, span = Pv(), Pv(), Pv()
blob = ct.create_string_buffer(marker.image)
device.check(device.load(ct.byref(module), ct.cast(blob, Pv)))
device.check(device.function(ct.byref(fn), module, marker.entry.encode()))
host_span = (ct.c_uint64*2)()
device.check(device.alloc(ct.byref(span), ct.sizeof(host_span)))
margv = (Pv*1)(ct.cast(ct.byref(span), Pv))
e0, e1 = Pv(), Pv()
device.check(device.event_create(ct.byref(e0), 0)); device.check(device.event_create(ct.byref(e1), 0))

def window(gap, markers, idle=False):
    if gap:
        host_span[0], host_span[1] = (1 << 64) - 1, 0
        device.check(device.copy(span, ct.addressof(host_span), ct.sizeof(host_span), 1))
        device.check(device.sync())
    if idle:
        time.sleep(0.002)
    device.check(device.event_record(e0, None))
    if markers: device.check(device.launch(fn,1,1,1,1,1,1,0,None,margv,None))
    for _ in range(100):
        device.check(device.launch(binding._bound._function,*grid,*block,shared,None,argv,None))
    if markers: device.check(device.launch(fn,1,1,1,1,1,1,0,None,margv,None))
    device.check(device.event_record(e1, None)); device.check(device.event_sync(e1))
    ms = ct.c_float(); device.check(device.event_elapsed(ct.byref(ms), e0, e1))
    return ms.value * 1e6 / 100

protocols = dict(A_plain=(False,False,False), B_plain_gap=(True,False,False), C_markers=(False,True,False),
                 D_recorder=(True,True,False), E_plain_idle=(False,False,True))
for idle_s in (0.0, 3.0, 3.0):
    for name, (gap, mk) in [('A_plain', (False, False)), ('D_recorder', (True, True))]:
        time.sleep(idle_s)
        w = [window(gap, mk) for _ in range(7)]
        print(f"{'cooperative' if coop else 'serial'} idle={idle_s}s {name:10s} median={statistics.median(w):9.1f} windows={[round(x) for x in w]}", flush=True)
