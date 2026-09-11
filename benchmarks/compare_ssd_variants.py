#!/usr/bin/env python3
"""Fixed-count independent-process paired SSD measurements.

Resident event windows include submission gaps. Results cannot promote a route:
clock calibration and production selector binding remain separate gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tessera.compiler.ssd_performance import summarize  # noqa: E402

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shape',nargs=4,type=int,default=[32,2,16,4])
    parser.add_argument('--chunk',type=int,default=8)
    args = parser.parse_args()
    script = Path(__file__).with_name('record_ssd_gpu.py')
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    pairs = []
    with tempfile.TemporaryDirectory(prefix='ssd-paired-') as directory:
        for index in range(9):
            pair = {}
            order = ('serial','cooperative') if index%2 == 0 else ('cooperative','serial')
            for variant in order:
                path = Path(directory)/f'{index}-{variant}.json'
                command = [sys.executable,str(script),'--backend',args.backend,'--compiler',str(args.compiler),
                           '--output',str(path),'--shape',*map(str,args.shape),'--chunk',str(args.chunk),'--profile']
                if variant == 'cooperative':
                    command.append('--cooperative')
                subprocess.run(command,check=True,timeout=180)
                pair[variant] = json.loads(path.read_text())
            pairs.append(pair)
            print(f'completed pair {index+1}/9',flush=True)
    result = summarize(pairs)
    result.update(pairs=pairs,recorder_sha256=digest)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
