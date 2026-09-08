#!/usr/bin/env python3
"""Prove LLVM assertion execution in a subprocess, not just a CMake option."""
import argparse
import hashlib
import json
import resource
import shlex
import signal
import subprocess
import tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-config',required=True,type=Path)
    parser.add_argument('--cxx',default='c++')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    def config(*flags):
        return subprocess.check_output([str(args.llvm_config),*flags],text=True).strip()
    if config('--assertion-mode')!='ON':
        raise RuntimeError('owning LLVM does not enable assertions')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        source=root/'probe.cpp'
        source.write_text('#include "llvm/ADT/SmallVector.h"\nint main() { llvm::SmallVector<int, 1> empty; empty.pop_back(); }\n')
        executable=root/'probe'
        flags=shlex.split(config('--cxxflags','--ldflags','--libs','support','--system-libs'))
        subprocess.run([args.cxx,str(source),*flags,'-o',str(executable)],check=True)
        def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result=subprocess.run([str(executable)],capture_output=True,text=True,preexec_fn=no_core)
        if result.returncode!=-signal.SIGABRT or 'Assertion' not in result.stderr:
            raise RuntimeError('LLVM assertion probe did not abort at the contract')
        packet=dict(schema=1,assertion_mode='ON',version=config('--version'),
            llvm_config_sha256=hashlib.sha256(args.llvm_config.read_bytes()).hexdigest(),
            source=source.read_text(),compiler_flags=flags,returncode=result.returncode,
            stderr=result.stderr,scope='LLVM header contract; Tessera pass validation is separate')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(packet,indent=2)+'\n')


if __name__=='__main__':main()
