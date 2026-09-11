#!/usr/bin/env python3
"""Correlate a dedicated SSD capture with successful CUDA launch API records.

Only supports the recorder's single-artifact, one-warmup/700-launch envelope.
This attributes execution; it does not grant performance eligibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from tessera.compiler.profiler_cuda_window import read_nsys_device


def attribute(packet, path):
    if (packet.get('backend') != 'nvidia' or packet.get('execution') != 'native_gpu'
            or len(packet.get('rows', [])) != 1):
        raise ValueError('dedicated single-artifact native CUDA SSD packet required')
    device = read_nsys_device(path)
    if device['process_id'] != packet.get('process_id') or device['architecture'] != packet.get('architecture'):
        raise ValueError('capture process or architecture disagrees')
    row = packet['rows'][0]
    if len(row.get('device_event_ms', [])) != 7:
        raise ValueError('seven measured windows required')
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as c:
        kernels = c.execute('SELECT k.start,k.end,k.correlationId,k.globalPid,k.deviceId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
        calls = c.execute('SELECT r.start,r.end,r.correlationId,r.globalTid,r.returnValue,s.value '
                          'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
                          'WHERE s.value = "cuLaunchKernel"').fetchall()
    if len(kernels) != 701 or len(calls) != 701:
        raise ValueError('expected exactly 701 kernels and launch calls')
    by_key = {}
    for start, end, correlation, tid, status, name in calls:
        key = (tid >> 24, correlation)
        if key in by_key or status != 0 or end <= start:
            raise ValueError('ambiguous or unsuccessful launch API record')
        by_key[key] = (start, end)
    records = []
    for start, end, correlation, pid, gpu, name in kernels:
        key = (pid >> 24, correlation)
        api = by_key.pop(key, None)
        # Nsight global process/thread IDs encode the OS process above bit 24.
        if (api is None or key[0] & 0xffffff != packet['process_id'] or gpu != device['device']
                or name != 'product' or end <= start or start < api[0]):
            raise ValueError('kernel lacks a unique owning-process successful launch')
        records.append(dict(correlation=correlation, api_start_ns=api[0], api_end_ns=api[1],
                            kernel_start_ns=start, kernel_end_ns=end))
    if by_key:
        raise ValueError('unmatched launch records')
    return dict(schema=1, backend='nvidia', process_id=packet['process_id'], run_id=packet['run_id'],
                binding_digest=row['binding_digest'], image_sha256=row['image_sha256'],
                capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), capture_device=device,
                scope='dedicated_single_artifact_process', profiler_kernel_count=701,
                matched_launch_count=701, records=records, promotion_eligible=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--sqlite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(attribute(json.loads(args.packet.read_text()), args.sqlite), indent=2) + '\n')


if __name__ == '__main__':
    main()
