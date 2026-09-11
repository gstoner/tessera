#!/usr/bin/env python3
"""Replay real SSD artifacts and record the production selector's decision."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402
from tessera.compiler.ssd_performance import bind_measured_ssd  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    packet = comparison['pairs'][0]['serial']
    T,H,N,P = packet['shape']
    logical = lower_scheduled_ssd(T,H,N,P,packet['rows'][0]['chunk'],compiler=args.compiler)
    options = dict(compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=packet['backend'],chip=packet['architecture'])
    serial = materialize_ssd(logical,**options)
    cooperative = materialize_ssd(logical,cooperative=True,**options)
    bound,decision = bind_measured_ssd(serial,cooperative,comparison)
    try:
        result = dict(decision=asdict(decision),selected_binding=bound.package.binding_digest,
                      incumbent_binding=serial.package.binding_digest,candidate_binding=cooperative.package.binding_digest)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        bound.close()


if __name__ == '__main__':
    main()
