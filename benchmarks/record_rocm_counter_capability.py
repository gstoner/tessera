#!/usr/bin/env python3
"""Record exact-host counter discovery; absence is a blocker, never a zero count."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def classify(result):
    text = result['stdout'] + result['stderr']
    if result['returncode']:
        return 'probe_failed'
    if 'No pmc counters supported' in text or 'no counter metrics found' in text:
        return 'unsupported'
    # Keep discovery separate from successful per-dispatch counter collection.
    return 'unverified'


def probe(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=argv, returncode=-1, stdout='', stderr=str(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rocm-bin', type=Path, default=Path('/opt/rocm/bin'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    discovery = probe([str(args.rocm_bin / 'rocprofv3-avail'), 'list', '--pmc'])
    data = dict(schema='tessera.rocm.counter_capability.v1', host=platform.node(),
        platform=platform.platform(), discovery=discovery, status=classify(discovery),
        version=probe([str(args.rocm_bin / 'rocprofv3'), '--version']),
        agents=probe([str(args.rocm_bin / 'rocminfo')]),
        alternatives=probe([str(args.rocm_bin / 'rocprofv3-avail'), 'list', '--spm', '--pc-sampling', '--spm-config']),
        device_counter_attribution=False,
        next_step='collect per-dispatch counters on a profiler-supported owning host; HIP API traces are not counters',
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(data['status'])


if __name__ == '__main__':
    main()
