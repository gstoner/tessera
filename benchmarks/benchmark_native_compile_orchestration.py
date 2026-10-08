"""A/B separate Python-driven processes versus one native MLIR pass manager."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from tessera.compiler import native_attention_jvp as jvp
from tessera.compiler import native_gpu_storage as storage
from tessera.compiler import toolchain_identity
from tessera.compiler.scheduled_matmul import run_tessera_opt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--llvm-bin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidate=jvp._lower_graph
    def split(graph,*,compiler=None):
        tool=Path(compiler)
        schedule=run_tessera_opt(tool,graph,'--tessera-graph-to-schedule')
        return run_tessera_opt(tool,schedule,'--tessera-schedule-to-tile')
    rows=[]
    run,sha=storage._run,storage._sha
    content_digest=storage.binary_content_digest
    for sk in (5,129):
        reference=None
        for trial in range(5):
            for arm in (('split','native_cold','native') if trial%2==0 else ('native','native_cold','split')):
                if arm=='native_cold':
                    toolchain_identity._FILE_DIGESTS.clear()
                events=[]
                def traced_run(tool,*options,**kw):
                    start=time.perf_counter()
                    try:
                        return run(tool,*options,**kw)
                    finally:
                        events.append(dict(kind='tool',tool=str(tool),options=list(options),
                                           wall_ms=(time.perf_counter()-start)*1000))
                def traced_sha(data):
                    start=time.perf_counter()
                    try:
                        return sha(data)
                    finally:
                        events.append(dict(kind='hash',bytes=len(data),wall_ms=(time.perf_counter()-start)*1000))
                def traced_content(path):
                    start=time.perf_counter()
                    try:
                        return (sha(path.read_bytes()) if arm=='split' else content_digest(path))
                    finally:
                        events.append(dict(kind='tool_identity',path=str(path),
                            wall_ms=(time.perf_counter()-start)*1000))
                storage.binary_content_digest=traced_content
                storage._run,storage._sha=traced_run,traced_sha
                jvp._lower_graph=split if arm=='split' else candidate
                start=time.perf_counter()
                try:
                    package=jvp.materialize((1,2,1,3,sk,4,3),.5,True,
                        compiler=args.compiler,llvm_bin=args.llvm_bin)
                finally:
                    jvp._lower_graph=candidate
                    storage._run,storage._sha=run,sha
                    storage.binary_content_digest=content_digest
                elapsed=(time.perf_counter()-start)*1000
                parity=(package.arena_ir,package.image,package.abi,package.entry,package.sizer)
                if reference is None:
                    reference=parity
                assert parity==reference
                rows.append(dict(sk=sk,trial=trial,arm=arm,wall_ms=elapsed,events=events,
                    image_sha256=hashlib.sha256(package.image).hexdigest(),
                    arena_sha256=hashlib.sha256(package.arena_ir.encode()).hexdigest()))
    args.output.write_text(json.dumps(dict(rows=rows,native_image_and_arena_parity=True,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        compiler_bytes=args.compiler.stat().st_size,
        source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('python/tessera/compiler/native_attention_jvp.py','python/tessera/compiler/native_gpu_storage.py','python/tessera/compiler/toolchain_identity.py')},
        comparison='split pass processes plus uncached exact tool hashes versus one native pass manager plus stable-file exact digest reuse; CUDA image validation unchanged',
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    for sk in (5,129):
        for arm in ('split','native_cold','native'):
            print(sk,arm,statistics.median(r['wall_ms'] for r in rows if r['sk']==sk and r['arm']==arm))
    print('largest stage events',sorted(rows[-1]['events'],key=lambda e:e['wall_ms'],reverse=True)[:4])


if __name__=='__main__':
    main()
