#!/usr/bin/env python3
"""SuperBench adapter for replay-bound native serial/cooperative SSD packages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    p.add_argument('--compiler', type=Path, default=os.environ.get('TESSERA_OPT'))
    p.add_argument('--shape', type=int, nargs=4, default=[32, 2, 8, 4])
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--cooperative', action='store_true')
    args = p.parse_args()
    if args.compiler is None:
        p.error('--compiler or TESSERA_OPT is required')
    if any(v <= 0 for v in args.shape) or args.chunk <= 0:
        p.error('shape and chunk must be positive')
    with tempfile.TemporaryDirectory(prefix='ssd-superbench-') as temp:
        output = Path(temp) / 'packet.json'
        command = [sys.executable, str(ROOT / 'benchmarks/record_ssd_gpu.py'),
                   '--backend', args.backend, '--compiler', str(args.compiler.resolve()),
                   '--shape', *map(str, args.shape), '--chunk', str(args.chunk),
                   '--profile', '--output', str(output)]
        if args.cooperative:
            command.append('--cooperative')
        subprocess.run(command, check=True)
        packet = json.loads(output.read_text())
    row = packet['rows'][0]
    if packet['execution'] != 'native_gpu' or len(row['device_event_ms']) != 7:
        raise ValueError('SSD adapter requires checked native execution and seven event windows')
    packet.update(ok=True, runtime_status='executable', execution_kind='native_gpu',
                  latency_ms=row['checked_call_ms'],
                  timing_domain='checked_resident_host_call',
                  device_event_ms=row['device_event_median_ms'],
                  device_timing_domain='resident_100_launch_event_window_per_call')
    print(json.dumps(packet))


if __name__ == '__main__':
    main()
