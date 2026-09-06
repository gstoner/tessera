#!/usr/bin/env python3
"""Fixed-count independent-process ROCm wait ablation; never a promotion gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(packets):
    if len(packets) < 5:
        raise ValueError('at least five independent processes required')
    if len({p['process_id'] for p in packets}) != len(packets):
        raise ValueError('cross-run packets must come from distinct processes')
    identity = ('compiler_sha256', 'source_sha256', 'recorder_sha256', 'images', 'device')
    for packet in packets:
        if any(packet[k] != packets[0][k] for k in identity):
            raise ValueError('cross-run compiler/source/image/device identity differs')
        if len(packet['rows']) != len(packets[0]['rows']):
            raise ValueError('cross-run workloads differ')
    results = []
    for i, first in enumerate(packets[0]['rows']):
        shape = {k: first[k] for k in ('blocks', 'width', 'rounds')}
        ratios = []
        matched_ratios = []
        for packet in packets:
            row = packet['rows'][i]
            if any(row[k] != v for k, v in shape.items()) or row['oracle'] != 'exact':
                raise ValueError('cross-run shape/oracle differs')
            for mode in ('prefetch', 'immediate_wait', 'nonblocking_wait'):
                samples = row['hip_event_ms'][mode]
                median = row['median_ms'][mode]
                if len(samples) != packet['trials'] or not samples or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples
                ) or type(median) not in (int, float) or median != statistics.median(samples):
                    raise ValueError('invalid or inconsistent device-event samples')
            ratios.append(row['median_ms']['immediate_wait'] / row['median_ms']['prefetch'])
            matched_ratios.append(row['median_ms']['immediate_wait'] / row['median_ms']['nonblocking_wait'])
        results.append(dict(**shape, control_over_prefetch=ratios,
            median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
            runs_favoring_prefetch=sum(r > 1 for r in ratios),
            control_over_nonblocking=matched_ratios, matched_median_ratio=statistics.median(matched_ratios),
            matched_runs_favoring_nonblocking=sum(r > 1 for r in matched_ratios)))
    return results


def matched_instructions(drain, nonblocking):
    def instructions(text):
        return [line.split('//')[0].strip() for line in text.splitlines() if '//' in line]
    a,b=instructions(drain),instructions(nonblocking)
    if not a or len(a)!=len(b):
        raise ValueError('matched controls have different instruction counts')
    differences=[(i,x,y) for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    if not differences or any(x!='s_waitcnt vmcnt(0)' or y!='s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63)' for _,x,y in differences):
        raise ValueError('matched controls differ beyond VMEM wait thresholds')
    return dict(instruction_count=len(a), changed_wait_indices=[i for i,_,_ in differences],
                all_other_instructions_identical=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    args = p.parse_args()
    if args.runs < 5:
        p.error('at least five fresh processes required')
    args.artifacts.mkdir(parents=True, exist_ok=True)
    recorder = Path(__file__).with_name('measure_rocm_prefetch.py')
    packets = []
    for run in range(args.runs):
        output = args.artifacts / f'run-{run}.json'
        subprocess.run([sys.executable, str(recorder), '--compiler', str(args.compiler),
            '--artifacts', str(args.artifacts / f'run-{run}'), '--output', str(output),
            '--seed', str(run), '--trials', '11', '--launches', '50'], check=True,
            stdout=subprocess.DEVNULL)
        packets.append(json.loads(output.read_text()))
    isa = matched_instructions((args.artifacts/'run-0/immediate_wait.disasm').read_text(),
                               (args.artifacts/'run-0/nonblocking_wait.disasm').read_text())
    report = dict(matched_isa=isa, processes=args.runs, selector_promotion=False,
        interpretation='Includes instruction-count-matched vmcnt(63) control. Verify disassembly/resource parity; timing alone is not a hardware-counter causality proof.',
        summaries=summarize(packets), packets=packets,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.artifacts / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summaries'], indent=2))


if __name__ == '__main__':
    main()
