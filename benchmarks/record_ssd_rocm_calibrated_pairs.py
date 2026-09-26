#!/usr/bin/env python3
"""Nine ROCm (gfx1151 or gfx1201) SSD process pairs, each process carrying its own device-clock
calibration, then the production SSD admission decision on the result.

Every process measures its clean serial or cooperative image with HIP events
(the comparison row) and calibrates that image against its compiler-built
``--tessera-device-clock-span`` twin (the calibration packet): the in-kernel
constant-rate device clock must agree with the HIP event it runs under, and
the instrumented/clean duration ratio is the overhead gate. On WSL2 this is
the ``device_clock_witness`` route (sync WSL-TIMING-ADMISSION-2026-09-26);
no rocprofiler or /dev/kfd is needed.

Qualification is checked before measurement: a dirty tree is refused unless
--diagnostic, and --diagnostic output can never be promotion evidence because
the packets record the dirty state and admission refuses it.

The chip is never assumed: each child process queries the active HIP device
and records it, and admission materializes the incumbent/candidate for the
chip the rows name (all eighteen must agree -- ``summarize`` refuses a mixed
identity; sync GFX1201-SSD-CALIBRATION-2026-09-26).
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.profiler_rocm_evidence import ROCM_PROFILER_ARCHITECTURES  # noqa: E402
from tessera.compiler.ssd_performance import bind_measured_ssd, summarize  # noqa: E402



def _llvm_bin():
    """The matched LLVM 23 bin directory (``TESSERA_LLVM_BIN`` first), never a
    hard-coded apt path: Tajasarus has no /usr/lib/llvm-23."""
    from tessera.compiler.llvm_tools import llvm_bin_dir
    found = llvm_bin_dir()
    if found is None:
        raise SystemExit('matched LLVM 23 tools not found (set TESSERA_LLVM_BIN)')
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--shape', type=int, nargs=4, default=(512, 2, 32, 8))
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--diagnostic', action='store_true')
    args = p.parse_args()

    def git(*cmd):
        return subprocess.check_output(['git', *cmd], cwd=ROOT, text=True).strip()

    source = dict(source_commit=git('rev-parse', 'HEAD'),
                  worktree_dirty=bool(git('status', '--porcelain')),
                  execution_environment='wsl2' if 'microsoft' in platform.release().lower() else 'bare_metal')
    if source['worktree_dirty'] and not args.diagnostic:
        raise SystemExit('source tree has uncommitted changes; commit first or pass --diagnostic')
    destination = args.output_dir.resolve()
    if destination.is_relative_to(ROOT):
        raise SystemExit('write measurement evidence outside the source checkout')
    destination.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python')}
    recorder = ROOT / 'benchmarks' / 'record_ssd_gpu.py'
    base = [sys.executable, str(recorder), '--backend', 'rocm', '--compiler', str(args.compiler.resolve()),
            '--shape', *map(str, args.shape), '--chunk', str(args.chunk), '--profile']

    pairs, calibrations = [], []
    for index in range(9):
        pair, calibrated = {}, {}
        names = ('serial', 'cooperative') if index % 2 == 0 else ('cooperative', 'serial')
        for name in names:
            row, packet = destination / f'{index}-{name}.json', destination / f'{index}-{name}-calibration.json'
            command = base + (['--cooperative'] if name == 'cooperative' else []) + [
                '--output', str(row), '--device-clock-calibration', str(packet)]
            subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=900)
            pair[name] = json.loads(row.read_text())
            calibrated[name] = json.loads(packet.read_text())
        pairs.append(pair)
        calibrations.extend(calibrated[name] for name in ('serial', 'cooperative'))
        print(f'completed pair {index + 1}/9', flush=True)
    if git('rev-parse', 'HEAD') != source['source_commit'] or bool(git('status', '--porcelain')) != source['worktree_dirty']:
        raise SystemExit('source state changed while collecting evidence')

    comparison = dict(pairs=pairs, calibrations=calibrations, source=source)
    comparison.update(summarize(pairs))
    chip = comparison['identity'][1]
    if comparison['identity'][0] != 'rocm' or chip not in ROCM_PROFILER_ARCHITECTURES:
        raise SystemExit(f'measured device {chip!r} has no ROCm calibration route')
    (destination / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')

    T, H, N, P = args.shape
    logical = lower_scheduled_ssd(T, H, N, P, args.chunk, compiler=args.compiler)
    options = dict(compiler=args.compiler, llvm_bin=_llvm_bin(), backend='rocm', chip=chip)
    serial = materialize_ssd(logical, **options)
    cooperative = materialize_ssd(logical, cooperative=True, **options)
    bound, decision = bind_measured_ssd(serial, cooperative, comparison, calibrations)
    try:
        routes = sorted({c.get('admission_route') for c in calibrations})
        result = dict(decision=asdict(decision), admission_routes=routes,
                      selected_binding=bound.package.binding_digest,
                      incumbent_binding=serial.package.binding_digest,
                      candidate_binding=cooperative.package.binding_digest, source=source)
        (destination / 'admission.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['decision'], indent=2))
    finally:
        bound.close()


if __name__ == '__main__':
    main()
