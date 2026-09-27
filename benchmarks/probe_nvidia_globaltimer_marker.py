#!/usr/bin/env python3
"""Validate the NVIDIA ``%globaltimer`` device-clock marker on the owning GPU.

Sync ``NVIDIA-GLOBALTIMER-MARKER-2026-09-26`` (follows
``DEVICE-CLOCK-MARKER-2026-09-26``). The marker is the compiler-built empty
kernel of :mod:`tessera.compiler.native_device_clock`; this probe settles the
three claims its admission needs, on this device and nowhere else:

1. **It builds and reads the clock twice into one span** --
   ``build_device_clock_marker(backend='nvidia', chip='sm_120')`` (which itself
   refuses an image whose SASS lacks two ``SR_GLOBALTIMERLO`` reads and two
   64-bit span atomics), then one marker launch must leave
   ``span[0] <= span[1]``, both written.
2. **Resolution.** A one-thread kernel reads ``%globaltimer`` back to back and
   stores every value; the distinct non-zero increments are the counter's
   update granularity as this driver exposes it. WSL2's may be coarse, which
   bounds how short a window the marker can time.
3. **Agreement with CUDA events over windows of varying length.** Each window
   is: reset the span, synchronize, event, marker, K launches of a spin kernel,
   marker, event. The span (first marker's start to second marker's end) and
   the event interval bracket the same stream interval by independent clocks;
   ``|event - span| / event`` must stay within the 5% band
   (``profiler_timing.CLOCK_AGREEMENT_BAND``) for windows long enough to use.

Every kernel here is compiled through ``tessera-opt``/``mlir-opt`` (NVVM ->
``gpu-module-to-binary``), not NVRTC or nvcc. Diagnostic evidence only: it
admits nothing by itself; the SSD calibrated-pairs packet is the admission
evidence. Run under ``flock /tmp/tessera-timing.lock`` on The-Super-Bear.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_device_clock import build_device_clock_marker  # noqa: E402
from tessera.compiler.native_gpu_storage import _decode_image, _run  # noqa: E402
from tessera.compiler.profiler_timing import CLOCK_AGREEMENT_BAND  # noqa: E402

_PROBES = """module attributes {gpu.container_module} {
  gpu.module @probe {
    gpu.func @sample(%out: !llvm.ptr<1>, %n: i64) kernel {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      scf.for %i = %c0 to %n step %c1 : i64 {
        %t = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
        %p = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
        llvm.store %t, %p : i64, !llvm.ptr<1>
      }
      gpu.return
    }
    gpu.func @spin(%out: !llvm.ptr<1>, %iters: i64) kernel {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %one = arith.constant 1.0 : f32
      %a = arith.constant 1.0000001 : f32
      %b = arith.constant 1.0e-7 : f32
      %acc = scf.for %i = %c0 to %iters step %c1 iter_args(%x = %one) -> (f32) : i64 {
        %y = llvm.intr.fma(%x, %a, %b) : (f32, f32, f32) -> f32
        scf.yield %y : f32
      }
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %base = arith.muli %bid, %bdim : index
      %g = arith.addi %base, %tid : index
      %gi = arith.index_cast %g : index to i64
      %p = llvm.getelementptr %out[%gi] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %acc, %p : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
"""

SPIN_BLOCKS, SPIN_THREADS = 48, 128


def _compile_probes(llvm_bin: Path) -> bytes:
    pipeline = ('builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm,'
                'convert-math-to-llvm,reconcile-unrealized-casts),'
                'nvvm-attach-target{chip=sm_120},gpu-module-to-binary)')
    binary = _run(llvm_bin / 'mlir-opt', '--pass-pipeline=' + pipeline, source=_PROBES)
    encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
    return _decode_image(encoded[1] if encoded else re.findall(r'"((?:\\.|[^"\\])*)"', binary)[-1])


def _identity(device: Device) -> dict:
    lib = device.lib
    dev = ct.c_int()
    device.check(lib.cuCtxGetDevice(ct.byref(dev)))
    name = ct.create_string_buffer(256)
    device.check(lib.cuDeviceGetName(name, 256, dev))
    major, minor = ct.c_int(), ct.c_int()
    device.check(lib.cuDeviceGetAttribute(ct.byref(major), 75, dev))
    device.check(lib.cuDeviceGetAttribute(ct.byref(minor), 76, dev))
    driver = ct.c_int()
    device.check(lib.cuDriverGetVersion(ct.byref(driver)))
    uuid = (ct.c_ubyte * 16)()
    device.check(lib.cuDeviceGetUuid_v2(uuid, dev))
    return {'name': name.value.decode(), 'ordinal': dev.value,
            'compute_capability': f'{major.value}.{minor.value}',
            'architecture': f'sm_{major.value}{minor.value}',
            'driver_api_version': driver.value, 'uuid': bytes(uuid).hex()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=7)
    args = parser.parse_args()
    from tessera.compiler.llvm_tools import llvm_bin_dir
    llvm_bin = llvm_bin_dir()
    if llvm_bin is None:
        raise SystemExit('matched LLVM 23 tools not found (set TESSERA_LLVM_BIN)')

    device = Device('nvidia')
    identity = _identity(device)
    if identity['architecture'] != 'sm_120':
        raise SystemExit(f"this probe validates sm_120; the device is {identity}")
    marker = build_device_clock_marker(compiler=args.compiler, llvm_bin=llvm_bin,
                                       backend='nvidia', chip='sm_120')
    probes = _compile_probes(llvm_bin)
    P = ct.c_void_p
    mods = [P(), P()]
    fns = {name: P() for name in ('marker', 'sample', 'spin')}
    marker_blob, probe_blob = ct.create_string_buffer(marker.image), ct.create_string_buffer(probes)
    device.check(device.load(ct.byref(mods[0]), ct.cast(marker_blob, P)))
    device.check(device.load(ct.byref(mods[1]), ct.cast(probe_blob, P)))
    device.check(device.function(ct.byref(fns['marker']), mods[0], marker.entry.encode()))
    device.check(device.function(ct.byref(fns['sample']), mods[1], b'sample'))
    device.check(device.function(ct.byref(fns['spin']), mods[1], b'spin'))

    span, samples, sink = P(), P(), P()
    n_samples = 65536
    device.check(device.alloc(ct.byref(span), 16))
    device.check(device.alloc(ct.byref(samples), 8 * n_samples))
    device.check(device.alloc(ct.byref(sink), 4 * SPIN_BLOCKS * SPIN_THREADS))
    host_span = (ct.c_uint64 * 2)()

    def reset_span():
        host_span[0], host_span[1] = (1 << 64) - 1, 0
        device.check(device.htod(span, ct.addressof(host_span), 16))
        device.check(device.sync())

    def read_span():
        device.check(device.dtoh(ct.addressof(host_span), span, 16))
        return int(host_span[0]), int(host_span[1])

    marker_argv = (P * 1)(ct.cast(ct.byref(span), P))

    # 1. The marker writes one ordered span.
    reset_span()
    device.check(device.launch(fns['marker'], 1, 1, 1, 1, 1, 1, 0, None, marker_argv, None))
    device.check(device.sync())
    single = read_span()
    if single[0] == (1 << 64) - 1 or single[1] < single[0]:
        raise SystemExit(f'one marker launch did not write an ordered span: {single}')

    # 2. Resolution: back-to-back reads in one thread.
    count = ct.c_int64(n_samples)
    sample_argv = (P * 2)(ct.cast(ct.byref(samples), P), ct.cast(ct.byref(count), P))
    reads = []
    for _ in range(3):
        device.check(device.launch(fns['sample'], 1, 1, 1, 1, 1, 1, 0, None, sample_argv, None))
        device.check(device.sync())
        host = (ct.c_uint64 * n_samples)()
        device.check(device.dtoh(ct.addressof(host), samples, 8 * n_samples))
        reads.append(list(host))
    resolution = []
    for values in reads:
        deltas = [b - a for a, b in zip(values, values[1:])]
        if any(d < 0 for d in deltas):
            raise SystemExit('%globaltimer went backwards within one thread')
        nonzero = [d for d in deltas if d]
        changes = len(nonzero)
        elapsed = values[-1] - values[0]
        histogram: dict[int, int] = {}
        for d in nonzero:
            histogram[d] = histogram.get(d, 0) + 1
        resolution.append({
            'reads': len(values), 'elapsed_ns': elapsed, 'changes': changes,
            'zero_delta_fraction': 1 - changes / len(deltas),
            'min_nonzero_delta_ns': min(nonzero) if nonzero else None,
            'median_nonzero_delta_ns': statistics.median(nonzero) if nonzero else None,
            'max_delta_ns': max(deltas),
            'mean_update_period_ns': elapsed / changes if changes else None,
            'ns_per_read': elapsed / (len(values) - 1),
            'top_nonzero_deltas': sorted(histogram.items(), key=lambda kv: -kv[1])[:8],
        })

    # 3. Agreement over windows of varying length.
    events = [P(), P()]
    for event in events:
        device.check(device.event_create(ct.byref(event), 0))
    iters = ct.c_int64(0)
    spin_argv = (P * 2)(ct.cast(ct.byref(sink), P), ct.cast(ct.byref(iters), P))
    configs = [(k, it) for it in (64, 4096, 65536) for k in (1, 10, 100, 1000)]
    windows = []
    for launches, iterations in configs:
        iters.value = iterations
        # Warm this configuration so its first timed window is not a cold start.
        for _ in range(3):
            device.check(device.launch(fns['spin'], SPIN_BLOCKS, 1, 1, SPIN_THREADS, 1, 1, 0,
                                       None, spin_argv, None))
        rows = []
        for _ in range(args.windows):
            reset_span()
            start = time.perf_counter_ns()
            device.check(device.event_record(events[0], None))
            device.check(device.launch(fns['marker'], 1, 1, 1, 1, 1, 1, 0, None, marker_argv, None))
            for _ in range(launches):
                device.check(device.launch(fns['spin'], SPIN_BLOCKS, 1, 1, SPIN_THREADS, 1, 1, 0,
                                           None, spin_argv, None))
            device.check(device.launch(fns['marker'], 1, 1, 1, 1, 1, 1, 0, None, marker_argv, None))
            device.check(device.event_record(events[1], None))
            device.check(device.event_sync(events[1]))
            host_ns = time.perf_counter_ns() - start
            ms = ct.c_float()
            device.check(device.event_elapsed(ct.byref(ms), events[0], events[1]))
            lo, hi = read_span()
            if lo == (1 << 64) - 1 or hi <= lo:
                raise SystemExit(f'marker span not written for {launches}x{iterations}: {(lo, hi)}')
            event_ns, span_ns = ms.value * 1e6, hi - lo
            rows.append({'event_ns': event_ns, 'span_ns': span_ns, 'host_ns': host_ns,
                         'event_minus_span_ns': event_ns - span_ns,
                         'relative_error': abs(event_ns - span_ns) / event_ns})
        errors = [r['relative_error'] for r in rows]
        windows.append({
            'launches': launches, 'spin_iterations': iterations,
            'median_event_ns': statistics.median(r['event_ns'] for r in rows),
            'median_span_ns': statistics.median(r['span_ns'] for r in rows),
            'median_host_ns': statistics.median(r['host_ns'] for r in rows),
            'median_event_minus_span_ns': statistics.median(r['event_minus_span_ns'] for r in rows),
            'median_relative_error': statistics.median(errors), 'max_relative_error': max(errors),
            'within_band_all_windows': all(e <= CLOCK_AGREEMENT_BAND for e in errors),
            'rows': rows,
        })
    for event in events:
        device.check(device.event_destroy(event))
    for pointer in (sink, samples, span):
        device.check(device.free(pointer))
    for module in mods:
        device.check(device.unload(module))

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip())
    result = {
        'schema': 'tessera.nvidia_globaltimer_marker_probe.v1',
        'sync_key': 'NVIDIA-GLOBALTIMER-MARKER-2026-09-26',
        'host': platform.node(), 'kernel_release': platform.release(),
        'execution_environment': 'wsl2' if 'microsoft' in platform.release().lower() else 'bare_metal',
        'device_identity': identity, 'source_commit': head, 'worktree_dirty': dirty,
        'marker_image_sha256': marker.image_sha256, 'single_marker_span': list(single),
        'agreement_band': CLOCK_AGREEMENT_BAND, 'spin_geometry': [SPIN_BLOCKS, SPIN_THREADS],
        'resolution': resolution, 'windows': windows,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'resolution': [{k: v for k, v in r.items() if k != 'top_nonzero_deltas'}
                                     for r in resolution],
                      'windows': [{k: w[k] for k in ('launches', 'spin_iterations', 'median_event_ns',
                                                     'median_span_ns', 'median_event_minus_span_ns',
                                                     'median_relative_error', 'max_relative_error',
                                                     'within_band_all_windows')}
                                  for w in windows]}, indent=1))


if __name__ == '__main__':
    main()
