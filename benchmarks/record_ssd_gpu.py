#!/usr/bin/env python3
"""Owning-device numerical proof for the replay-bound serial SSD baseline."""
import argparse
import os
import uuid
import ctypes as ct
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402

import platform  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

WINDOWS, LAUNCHES = 7, 100


def _llvm_bin():
    """The matched LLVM 23 bin directory (``TESSERA_LLVM_BIN`` first), never a
    hard-coded apt path: Tajasarus has no /usr/lib/llvm-23."""
    from tessera.compiler.llvm_tools import llvm_bin_dir
    found = llvm_bin_dir()
    if found is None:
        raise SystemExit('matched LLVM 23 tools not found (set TESSERA_LLVM_BIN)')
    return found


def _hip_enum(name):
    """An enum value from THIS host's HIP headers, never a hard-coded guess."""
    root = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    include = root / 'include'
    if not (include / 'hip' / 'hip_runtime_api.h').is_file():
        raise SystemExit(f'device-clock calibration needs HIP headers under {include} '
                         '(source scripts/_rocm_env.sh)')
    with tempfile.TemporaryDirectory(prefix='hip-enum-') as tmp:
        src, exe = Path(tmp)/'probe.c', Path(tmp)/'probe'
        src.write_text('#include <hip/hip_runtime_api.h>\n#include <stdio.h>\n'
                       f'int main(void){{printf("%d",(int){name});return 0;}}\n')
        subprocess.run(['cc','-D__HIP_PLATFORM_AMD__','-I',str(include),str(src),'-o',str(exe)],
                       check=True,capture_output=True,timeout=60)
        return int(subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=30).stdout)


def _isa_sha256(image, llvm_bin):
    with tempfile.TemporaryDirectory(prefix='ssd-isa-') as tmp:
        obj = Path(tmp)/'image.hsaco'
        obj.write_bytes(image)
        text = subprocess.run([str(llvm_bin/'llvm-objdump'),'-d',str(obj)],check=True,
                              capture_output=True,text=True,timeout=120).stdout
    body = '\n'.join(line for line in text.splitlines() if not line.startswith(str(obj)))
    return hashlib.sha256(body.encode()).hexdigest()


def _resources(device, function):
    out = {}
    for key, attribute in (('lds_bytes', 1), ('scratch_bytes', 3), ('vgpr', 4)):
        value = ct.c_int()
        device.check(device.attribute(ct.byref(value), attribute, function))
        out[key] = value.value
    return out


def _rocm_identity(device):
    """The active HIP device's identity, queried -- never the requested target.

    Refuses any architecture without a device-clock calibration route; the
    chip threads through materialization, the marker, the timing target and
    the image records, so one process cannot mix gfx1151 and gfx1201 facts.
    """
    from benchmarks.calibration.calibrate_gfx1151 import _active_device_identity
    from tessera.compiler.profiler_rocm_evidence import ROCM_PROFILER_ARCHITECTURES
    identity = _active_device_identity(device.lib)
    if identity['architecture'] not in ROCM_PROFILER_ARCHITECTURES:
        raise SystemExit(f"device-clock SSD calibration supports "
                         f"{', '.join(ROCM_PROFILER_ARCHITECTURES)}; this device is "
                         f"{identity['architecture']}")
    return identity


def device_clock_calibration(*, device, logical, clean_program, clean_binding, raw, grid, block,
                             clean_event_ms, reset, verify, compiler, llvm_bin, output, run_id):
    """Calibrate one process's clean SSD image with compiler-built device-clock markers.

    Each window: reset the span, record a HIP event, launch the marker, launch
    the EXACT clean image LAUNCHES times, launch the marker, record a HIP
    event. The two markers share the span buffer, so it runs from the first
    marker's start to the second marker's end on the device's constant-rate
    clock -- the same stream interval the HIP events bracket, measured
    independently of the event API. The clean image is never modified (an
    in-kernel stamp was measured to change its codegen; see
    tessera.compiler.native_device_clock). The marker-bracketed / plain event
    ratio is the overhead the packet bounds on both sides.
    """
    from tessera.compiler.native_device_clock import build_device_clock_marker
    from tessera.compiler.profiler_timing import (
        build_timing_sample, measured_clock, unavailable_clock, wall_clock_ticks_to_ns)
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet
    if device.cuda:
        raise SystemExit('device-clock calibration is implemented for ROCm here; '
                         'NVIDIA uses the Nsight activity-window recorder')
    # Query the device rather than trusting the requested target (review).
    identity = _rocm_identity(device)
    chip = identity['architecture']
    rate = ct.c_int()
    get_attribute = device.lib.hipDeviceGetAttribute
    get_attribute.argtypes, get_attribute.restype = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int
    device.check(get_attribute(ct.byref(rate), _hip_enum('hipDeviceAttributeWallClockRate'), 0))
    if rate.value <= 0:
        raise SystemExit('hipDeviceAttributeWallClockRate is not positive on this device')
    marker = build_device_clock_marker(compiler=compiler, llvm_bin=llvm_bin, backend='rocm', chip=chip)
    P = ct.c_void_p
    module, marker_fn, span = P(), P(), P()
    blob = ct.create_string_buffer(marker.image)
    host_span = (ct.c_uint64 * 2)()
    events = [P(), P()]
    try:
        device.check(device.load(ct.byref(module), ct.cast(blob, P)))
        device.check(device.function(ct.byref(marker_fn), module, marker.entry.encode()))
        device.check(device.alloc(ct.byref(span), ct.sizeof(host_span)))
        marker_argv = (P * 1)(ct.cast(ct.byref(span), P))
        values, shared = clean_binding._bound._launch_size(tuple(raw), grid, block)
        argv = (P*len(values))(*[ct.cast(ct.byref(v),P) for v in values])
        for event in events:
            device.check(device.event_create(ct.byref(event), 0))
        reset()  # outputs to NaN: verify() below then proves these windows computed them
        device_ns, event_ns, host_ns = [], [], []
        for _ in range(WINDOWS):
            host_span[0], host_span[1] = (1 << 64) - 1, 0
            device.check(device.copy(span, ct.addressof(host_span), ct.sizeof(host_span), 1))
            device.check(device.sync())
            start = time.perf_counter_ns()
            device.check(device.event_record(events[0], None))
            device.check(device.launch(marker_fn,1,1,1,1,1,1,0,None,marker_argv,None))
            for _ in range(LAUNCHES):
                device.check(device.launch(clean_binding._bound._function,*grid,*block,shared,None,argv,None))
            device.check(device.launch(marker_fn,1,1,1,1,1,1,0,None,marker_argv,None))
            device.check(device.event_record(events[1], None))
            device.check(device.event_sync(events[1]))
            host_ns.append((time.perf_counter_ns() - start) / LAUNCHES)
            ms = ct.c_float()
            device.check(device.event_elapsed(ct.byref(ms), events[0], events[1]))
            event_ns.append(ms.value * 1e6 / LAUNCHES)
            device.check(device.copy(ct.addressof(host_span), span, ct.sizeof(host_span), 2))
            if host_span[0] == (1 << 64) - 1 or host_span[1] <= host_span[0]:
                raise SystemExit(f'device-clock marker span was not written: {list(host_span)}')
            device_ns.append(wall_clock_ticks_to_ns(host_span[1] - host_span[0], rate.value) / LAUNCHES)
        verify()
        resources = _resources(device, clean_binding._bound._function)
    finally:
        # Run every release even if one fails, and never let a cleanup error
        # replace the exception that brought us here (review).
        failures = []
        for release in ([lambda e=e: device.event_destroy(e) for e in events if e.value]
                        + ([lambda: device.free(span)] if span.value else [])
                        + ([lambda: device.unload(module)] if module.value else [])):
            status = release()
            if status:
                failures.append(status)
        if failures and sys.exc_info()[0] is None:
            raise RuntimeError(f'device-clock calibration cleanup failed with status {failures}')
    environment = 'wsl2' if 'microsoft' in platform.release().lower() else 'bare_metal'
    sample_id = f'{output.stem}-{uuid.uuid4().hex[:12]}'
    clean_sha = hashlib.sha256(clean_program.package.image).hexdigest()
    semantic = hashlib.sha256(logical.schedule_ir.encode()).hexdigest()
    timing = build_timing_sample(
        sample_id=sample_id, target=f'rocm_{chip}',
        clocks={
            'host_wall_ns': measured_clock('host_wall_ns', source='perf_counter',
                                           value=statistics.median(host_ns)),
            'hip_event_ns': measured_clock('hip_event_ns', source='hip_event',
                                           value=statistics.median(event_ns)),
            'device_wall_clock_ns': measured_clock(
                'device_wall_clock_ns', source='device_wall_clock',
                value=statistics.median(device_ns), instrumented=True,
                calibrated_against=('hip_event_ns',), eligible_for_promotion=True,
                provenance={'method': 'compiler_built_marker_bracketing',
                            'clock': 'llvm.readsteadycounter', 'wall_clock_rate_khz': rate.value,
                            'windows': WINDOWS, 'launches_per_window': LAUNCHES,
                            'marker_image_sha256': marker.image_sha256,
                            'per_window_ns': device_ns}),
            'profiler_activity_ns': unavailable_clock(
                'profiler_activity_ns', source='rocprofiler_activity',
                reason='ROCPROFILER_UNAVAILABLE_NO_KFD' if environment == 'wsl2' else 'NOT_CAPTURED'),
        },
        artifact_digests={'application_image': clean_sha, 'device_clock_marker': marker.image_sha256,
                          'schedule_ir': semantic},
        batch_size=LAUNCHES, warm_state='warm', synchronization='hipEventSynchronize',
        execution_environment=environment, resources={'application': resources},
        environment={'kernel_release': platform.release(), 'per_window_event_ns': event_ns,
                     'per_window_host_ns': host_ns, 'run_id': run_id, 'process_id': os.getpid(),
                     'device_identity': identity})
    isa = _isa_sha256(clean_program.package.image, llvm_bin)
    image = dict(architecture=chip, kernel_name=clean_program.package.entry, semantic_sha256=semantic,
                 image_sha256=clean_sha, isa_sha256=isa, clock_source='hip_event',
                 calibration_sample_id=sample_id, resources=resources)
    # Same image both times: "instrumented" means measured under marker
    # bracketing, and its duration ratio is the markers' overhead.
    clean = dict(image, duration_ns=statistics.median(clean_event_ms) * 1e6, instrumented=False)
    probe = dict(image, duration_ns=statistics.median(event_ns), instrumented=True)
    capture = {
        'schema': 'tessera.profiler_rocm_native_capture.v1', 'provider': 'rocprofiler',
        'status': 'blocked', 'fresh_process': True, 'process': {'clean_exit': True},
        'reason': ('rocprofiler requires /dev/kfd; this WSL2 host exposes /dev/dxg only'
                   if environment == 'wsl2' else 'rocprofiler capture not requested'),
        'proof': {'dispatch_activity_seen': False, 'hip_callback_seen': False,
                  'counter_records_seen': False, 'pc_samples_seen': False},
        'requested': {'counters': [], 'pc_sampling': False},
        'provider_trace': _empty_trace(), 'eligible_for_promotion': False,
    }
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    dirty = bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip())
    packet = build_rocm_profiler_packet(timing=timing, capture=capture, uninstrumented=clean,
                                        instrumented=probe, source={'source_commit': head, 'worktree_dirty': dirty})
    output.write_text(json.dumps(packet, indent=2) + '\n')
    return packet


def _empty_trace():
    from tessera.compiler.profiler_provider_trace import build_provider_trace_artifact
    return build_provider_trace_artifact(provider='rocprofiler', records=(), source_status='unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cooperative',action='store_true')
    parser.add_argument('--shape',type=int,nargs=4,default=(5,2,3,2))
    parser.add_argument('--chunk',type=int)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--device-clock-calibration',type=Path,
                        help='ROCm: also calibrate the clean image with compiler-built '
                             'device-clock markers and write the packet here')
    args = parser.parse_args()
    if args.device_clock_calibration and not (args.profile and args.chunk and args.backend == 'rocm'):
        parser.error('--device-clock-calibration needs --backend rocm, --profile and one --chunk')
    run_id = uuid.uuid4().hex
    device = Device(args.backend)
    # The chip is the queried device's, never assumed (ROCm): it selects the
    # image, and the row records it for the admission identity check.
    chip = 'sm_120' if device.cuda else _rocm_identity(device)['architecture']
    rows = []
    T,H,N,P = args.shape
    for chunk in ((args.chunk,) if args.chunk else (1,2,5)):
        logical = lower_scheduled_ssd(T,H,N,P,chunk,compiler=args.compiler)
        program = materialize_ssd(logical,compiler=args.compiler,llvm_bin=_llvm_bin(),
                                  backend=args.backend,chip=chip,cooperative=args.cooperative)
        rng = np.random.default_rng(740+chunk)
        inputs = [rng.uniform(-.5,.5,shape).astype(np.float32)
                  for shape in [(T,H,P),(T,H),(T,H,N),(T,H,N),(H,N,P)]]
        x,decay,b,c,state = inputs
        state = state.copy()
        y = np.empty_like(x)
        saved = []
        for t in range(T):
            state = decay[t,:,None,None]*state+b[t,:,:,None]*x[t,:,None,:]
            y[t] = (c[t,:,:,None]*state).sum(axis=1)
            if (t+1)%chunk == 0 or t == T-1:
                saved.append(state.copy())
        expected = [y,state,np.array(saved)]
        outputs = [np.full_like(v,np.nan) for v in expected]
        pointers,views = [],[]
        bind_start = time.perf_counter()
        binding = program.bind()
        bind_ms = (time.perf_counter()-bind_start)*1000
        try:
            for value in inputs+outputs:
                pointer = ct.c_void_p()
                device.check(device.alloc(ct.byref(pointer),value.nbytes)); pointers.append(pointer)
                device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda
                             else device.copy(pointer,value.ctypes.data,value.nbytes,1))
                views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,
                             typestr=value.dtype.str,data=(pointer.value,False))))
            launch_start = time.perf_counter()
            binding(*views,1)
            checked_call_ms = (time.perf_counter()-launch_start)*1000
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
            timings = []
            if args.profile:
                # Resident direct launches exclude copies and Python descriptor
                # validation. Device events and end-to-end calls stay separate.
                raw,_,grid,block,_ = binding._resident(*views,1)
                values,shared = binding._bound._launch_size(raw,grid,block)
                argv = (ct.c_void_p*len(values))(*[ct.cast(ct.byref(v),ct.c_void_p) for v in values])
                start,end = ct.c_void_p(),ct.c_void_p()
                device.check(device.event_create(ct.byref(start),0))
                try:
                    device.check(device.event_create(ct.byref(end),0))
                    for _ in range(7):
                        device.check(device.event_record(start,None))
                        for _ in range(100):
                            device.check(device.launch(binding._bound._function,*grid,*block,shared,None,argv,None))
                        device.check(device.event_record(end,None)); device.check(device.event_sync(end))
                        ms = ct.c_float()
                        device.check(device.event_elapsed(ct.byref(ms),start,end))
                        timings.append(ms.value/100)
                finally:
                    if end.value: device.check(device.event_destroy(end))
                    device.check(device.event_destroy(start))
                if args.device_clock_calibration:
                    def reset():
                        for i,value in enumerate(outputs):
                            poison = np.full_like(value,np.nan)
                            device.check(device.copy(pointers[5+i],poison.ctypes.data,poison.nbytes,1))
                    def verify():
                        for i,value in enumerate(outputs):
                            result = np.empty_like(value)
                            device.check(device.copy(result.ctypes.data,pointers[5+i],result.nbytes,2))
                            np.testing.assert_allclose(result,expected[i],rtol=1e-5,atol=1e-6)
                    device_clock_calibration(
                        device=device, logical=logical, clean_program=program, clean_binding=binding,
                        raw=raw, grid=grid, block=block, clean_event_ms=timings, reset=reset, verify=verify,
                        compiler=args.compiler, llvm_bin=_llvm_bin(),
                        output=args.device_clock_calibration, run_id=run_id)
            rows.append(dict(chunk=chunk,binding_ms=bind_ms,checked_call_ms=checked_call_ms,device_event_ms=timings,
                             device_event_median_ms=statistics.median(timings) if timings else None,max_abs_errors=observed,binding_digest=program.package.binding_digest,
                             image_sha256=hashlib.sha256(program.package.image).hexdigest()))
        finally:
            binding.close()
            for pointer in reversed(pointers):
                device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(schema=1,backend=args.backend,process_id=os.getpid(),run_id=run_id,
        architecture=chip,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        shape=args.shape,clock='CUDA events' if device.cuda else 'HIP events',execution='native_gpu',cooperative=args.cooperative,rows=rows,promotion_eligible=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
