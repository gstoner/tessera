#!/usr/bin/env python3
"""CUDA owning-device proof of generation-sensitive fixed-slot arena reuse."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_gpu_storage import build_native_gpu_storage  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--slot-alias', action='store_true', help='measure a released two-slot permutation')
    p.add_argument('--pending-swap', action='store_true')
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--nested', action='store_true')
    p.add_argument('--outstanding', type=int, default=1)
    args = p.parse_args()
    if args.outstanding > 1 and (args.slot_alias or args.pending_swap or args.slots != 2):
        p.error('--outstanding selects its own matched slot/token protocol')
    if not 1 <= args.outstanding <= 4:
        p.error('outstanding count must be 1..4')
    if not 2 <= args.slots <= 8:
        p.error('slot count must be 2..8')
    fixture = 'tile_dynamic_gpu_slot_alias.mlir' if args.slot_alias else 'tile_dynamic_gpu_rotating.mlir'
    if args.pending_swap:
        fixture = 'tile_dynamic_gpu_pending_swap.mlir'
    source = (ROOT / 'tests/tessera-ir/phase3' / fixture).read_text()
    slots = 2 if args.slot_alias or args.pending_swap else 1
    if args.slot_alias and args.slots > 2:
        count = args.slots
        additions = ''.join(f'      %extra{i} = memref.alloca(%n) : memref<?xf32, 3>\n      "tile.alloc_shared"(%extra{i}) : (memref<?xf32, 3>) -> ()\n' for i in range(count-2))
        source = source.replace('      %slots:3', additions + '      %slots:' + str(count+1))
        source = source.replace('%acc = %initial)', '%acc = %initial, ' + ', '.join(f'%slot{i} = %extra{i}' for i in range(count-2)) + ')')
        source = source.replace('memref<?xf32, 3>, f32) {', 'memref<?xf32, 3>, f32' + ', memref<?xf32, 3>'*(count-2) + ') {')
        source = source.replace('scf.yield %write_slot, %read_slot, %total : memref<?xf32, 3>, memref<?xf32, 3>, f32',
            'scf.yield %write_slot, %slot0, %total, ' + ', '.join([f'%slot{i}' for i in range(1,count-2)] + ['%read_slot']) +
            ' : memref<?xf32, 3>, memref<?xf32, 3>, f32' + ', memref<?xf32, 3>'*(count-2))
        slots = count
    if args.pending_swap and args.slots > 2:
        extra = args.slots - 2
        allocations = ''.join(f'      %extra{i} = memref.alloca(%n) : memref<?xf32, 3>\n      "tile.alloc_shared"(%extra{i}) : (memref<?xf32, 3>) -> ()\n' for i in range(extra))
        source = source.replace('      %seed =', allocations + '      %seed =')
        source = source.replace('%last:4 =', f'%last:{args.slots+2} =')
        source = source.replace('%write_slot = %other)', '%write_slot = %other, ' + ', '.join(f'%slot{i} = %extra{i}' for i in range(extra)) + ')')
        source = source.replace('f32, memref<?xf32, 3>, memref<?xf32, 3>) {', 'f32, memref<?xf32, 3>, memref<?xf32, 3>' + ', memref<?xf32, 3>'*extra + ') {')
        source = source.replace('scf.yield %new_group, %total, %write_slot, %read_slot : !nvgpu.device.async.token, f32, memref<?xf32, 3>, memref<?xf32, 3>',
            'scf.yield %new_group, %total, %write_slot, ' + ', '.join([f'%slot{i}' for i in range(extra)] + ['%read_slot']) +
            ' : !nvgpu.device.async.token, f32' + ', memref<?xf32, 3>'*args.slots)
        slots = args.slots
    if args.outstanding > 1:
        from benchmarks.pending_storage_source import pending_cohort
        source = pending_cohort(args.outstanding)
        slots = 2 * args.outstanding
    if args.nested:
        source = source.replace('      %a = memref.alloca', '      %outerTrips = arith.constant 3 : index\n      scf.for %outer = %zero to %outerTrips step %one {\n      %a = memref.alloca')
        source = source.replace('      gpu.return', '      }\n      gpu.return')
    package = build_native_gpu_storage(source, compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                                       backend='nvidia', chip='sm_120')
    device = Device('nvidia')
    rows = []
    bound = package.bind()
    try:
        for width, rounds in [(32, 0), (32, 1), (32, 2), (64, 7), (128, 16), (128, 17), (256, 33)]:
            x = (np.arange(args.outstanding*(rounds+1)*32*width, dtype=np.float32) % 127).reshape(args.outstanding,rounds+1,32,width)
            y = np.zeros((32,width),dtype=np.float32)
            src, dst = ct.c_void_p(), ct.c_void_p()
            try:
                device.check(device.alloc(ct.byref(src), x.nbytes))
                device.check(device.alloc(ct.byref(dst), y.nbytes))
                device.check(device.htod(src, x.ctypes.data, x.nbytes))
                assert bound._size(src, dst, ct.c_int64(width), ct.c_int64(rounds)) == slots*width*4
                bound.launch((src.value, dst.value, width, rounds), grid=(32,1,1), block=(width,1,1))
                device.check(device.dtoh(y.ctypes.data, dst, y.nbytes))
                np.testing.assert_array_equal(y, np.roll(x[:, :-1], -1, axis=-1).sum(axis=(0, 1)))
                rows.append(dict(width=width, rounds=rounds, oracle='exact', native_bytes=slots*width*4,
                                 proof='successive generations use different input values; post-loop scratch reuses released slot'))
            finally:
                for ptr in [src,dst]:
                    if ptr.value:
                        device.check(device.free(ptr))
    finally:
        bound.close()
    args.output.write_text(json.dumps(dict(backend='nvidia', outstanding=args.outstanding, slot_alias=args.slot_alias, slots=slots, nested=args.nested, pending_swap=args.pending_swap or args.outstanding > 1, device=subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(), package_digest=package.binding_digest,
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows,
        selector_promotion=False),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
