#!/usr/bin/env python3
"""Attribute synchronous native package calls using NVTX plus CUDA correlations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def attribute(packet, path):
    if (packet.get('backend') != 'nvidia' or packet.get('nvtx') is not True
            or packet.get('execution_kind') != 'native_gpu' or packet.get('ok') is not True
            or type(packet.get('process_id')) is not int
            or not re.fullmatch('[0-9a-f]{32}', packet.get('run_id',''))):
        raise ValueError('successful NVTX native CUDA packet required')
    expected = {}
    for row in packet['rows']:
        if (row['label'] in expected or not re.fullmatch('[0-9a-f]{64}',row['artifact'])
                or not re.fullmatch('[0-9a-f]{64}',row['image'])
                or not row['label'].startswith(f'tessera:{packet["run_id"]}:{row["artifact"]}:{row["image"]}:')):
            raise ValueError('invalid or duplicate artifact range identity')
        expected[row['label']] = row
    if not expected:
        raise ValueError('empty range manifest')
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as c:
        ranges = c.execute('SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE "tessera:%"').fetchall()
        calls = c.execute('SELECT r.start,r.end,r.globalTid,r.correlationId,r.returnValue,s.value '
                          'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
                          'WHERE s.value = "cuLaunchKernel"').fetchall()
        kernels = c.execute('SELECT k.start,k.end,k.globalPid,k.correlationId,k.deviceId,k.contextId,k.streamId,s.value '
                            'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
    selected = {}
    for start,end,tid,label in ranges:
        if (tid >> 24) & 0xffffff != packet['process_id']:
            continue
        if label not in expected or label in selected or end is None or end <= start:
            raise ValueError('capture range disagrees with manifest')
        selected[label] = (start,end,tid)
    if selected.keys() != expected.keys():
        raise ValueError('missing artifact ranges')
    ordered = sorted(selected.values())
    if len({v[2] for v in ordered}) != 1 or any(a[1] > b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('overlapping or multi-thread ranges unsupported')
    records = {label:[] for label in expected}
    assigned = {}
    for start,end,tid,correlation,status,name in calls:
        owners = [label for label,(lo,hi,thread) in selected.items() if thread == tid and lo <= start and end <= hi]
        if not owners:
            continue  # Unmeasured warmups have no range and are not attributed.
        if len(owners) != 1 or status != 0 or end <= start:
            raise ValueError('ambiguous or failed launch in measured range')
        key = (tid >> 24,correlation)
        if key in assigned:
            raise ValueError('duplicate launch correlation')
        assigned[key] = (owners[0],start,end)
    for start,end,pid,correlation,device,context,stream,name in kernels:
        key = (pid >> 24,correlation)
        if key not in assigned:
            if ((pid >> 24) & 0xffffff == packet['process_id']
                    and any(lo <= start < hi for lo,hi,_ in selected.values())):
                raise ValueError('measured kernel lacks a launch correlation')
            continue
        label,api_start,api_end = assigned[key]
        lo,hi,_ = selected[label]
        if end <= start or start < api_start or end > hi:
            raise ValueError('kernel completion escapes synchronous range')
        records[label].append(dict(kernel=name,start_ns=start,end_ns=end,device=device,context=context,
                                   stream=stream,correlation=correlation,api_start_ns=api_start,api_end_ns=api_end))
    for key,(label,_,_) in assigned.items():
        if sum(r['correlation'] == key[1] for r in records[label]) != 1:
            raise ValueError('launch lacks exactly one correlated kernel')
    if any(not rows for rows in records.values()):
        raise ValueError('range has no correlated kernel')
    devices = {r['device'] for rows in records.values() for r in rows}
    if len(devices) != 1:
        raise ValueError('mixed-device capture unsupported')
    return dict(schema=1,process_id=packet['process_id'],run_id=packet['run_id'],
        capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),promotion_eligible=False,
        scope='synchronous_nvtx_calls',unmeasured_launches=len(calls)-len(assigned),
        rows=[dict(label=label,artifact=expected[label]['artifact'],image=expected[label]['image'],
                   workload=expected[label]['workload'],kernel_count=len(rows),kernels=rows)
              for label,rows in records.items()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','sqlite','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(attribute(json.loads(args.packet.read_text()),args.sqlite),indent=2)+'\n')


if __name__ == '__main__':
    main()
