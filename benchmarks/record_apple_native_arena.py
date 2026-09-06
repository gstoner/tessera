#!/usr/bin/env python3
"""Build and execute the exported native arena pair on the owning Apple host."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from tessera.compiler.apple_native_arena import AppleNativeArena


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('requires an owning Apple Silicon host')
    data=json.loads((args.artifacts/'artifact.json').read_text())
    digest=data.pop('digest')
    artifact=AppleNativeArena(**data)
    if artifact.digest != digest or (args.artifacts/'kernel.metal').read_text() != artifact.msl or (args.artifacts/'sizer.ll').read_text() != artifact.host_llvm_ir:
        raise ValueError('exported Apple shader/sizer identity disagrees')
    root=Path(__file__).resolve().parents[1]
    runner=root/'benchmarks/apple_gpu/record_dynamic_arena.mm'
    library=args.artifacts/'sizer.dylib'
    executable=args.artifacts/'record-arena'
    subprocess.run(['xcrun','clang','-dynamiclib','-O2',str(args.artifacts/'sizer.ll'),'-o',str(library)],check=True)
    subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-framework','Metal','-framework','Foundation',str(runner),'-o',str(executable)],check=True)
    result=json.loads(subprocess.check_output([str(executable),str(args.artifacts/'kernel.metal'),str(library),
                                              '_mlir_ciface_'+artifact.sizer],text=True,timeout=60))
    result.update(artifact_digest=digest, compiler_sha256=artifact.compiler_digest,
        msl_sha256=hashlib.sha256(artifact.msl.encode()).hexdigest(),
        llvm_sizer_sha256=hashlib.sha256(artifact.host_llvm_ir.encode()).hexdigest(),
        native_sizer_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        clang=subprocess.check_output(['xcrun','clang','--version'],text=True),
        boundary='owning-host materialization probe; no production selector or runtime dispatch integration')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['rows'],indent=2))


if __name__ == '__main__':
    main()
