"""Compare exact gfx1201 forward images; static ISA is not runtime attribution."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from tessera.compiler.rocm_native import _extract_hsaco
from benchmarks.rocm.benchmark_rocm_arch_fragments import _resources


def record(directory):
    directory.mkdir(parents=True, exist_ok=True)
    compiler = Path(os.environ['TESSERA_OPT'])
    objdump = Path(os.environ['TESSERA_LLVM_BIN']) / 'llvm-objdump'
    pipeline = ('builtin.module(generate-wmma-flash-attn-kernel,'
                'lower-tessera-target-to-rocdl,'
                'gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),'
                'rocdl-attach-target{chip=gfx1201},gpu-module-to-binary)')
    rows = []
    for half in (False, True):
        name = 'half' if half else 'baseline'
        source = ('module { "tessera_rocm.flash_attn"() {name = "attention_forward", '
                  'head_dim = 64 : i64, dtype = "f16", arch = "gfx1201", save_lse = true, '
                  f'half_fragment_loads = {str(half).lower()}' + '} : () -> () }\n')
        output = subprocess.check_output([str(compiler), '-', '--pass-pipeline='+pipeline],
                                         input=source, text=True)
        image = _extract_hsaco(output)
        path = directory / (name+'.hsaco')
        path.write_bytes(image)
        assembly = subprocess.check_output([str(objdump), '--disassemble', str(path)], text=True)
        if '<attention_forward>:' not in assembly:
            raise ValueError('emitted image lacks the expected kernel')
        # Mnemonics are before the address/encoding comment in AMD objdump.
        mnemonics = Counter(re.findall(r'^\s+([sv]_[a-z0-9_]+|(?:global|ds|flat)_[a-z0-9_]+)\b', assembly, re.M))
        if not any('wmma' in key for key in mnemonics):
            raise ValueError('image has no WMMA instruction witness')
        (directory / (name+'.mlir')).write_text(source)
        (directory / (name+'.s')).write_text(assembly)
        resources = _resources(output)
        if resources['vgpr_count'] is None:
            raise ValueError('compiler binary lacks kernel resource metadata')
        rows.append(dict(variant=name, kernel='attention_forward', resources=resources, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         image_sha256=hashlib.sha256(image).hexdigest(),
                         assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
                         instruction_counts=dict(sorted(mnemonics.items()))))
    packet = dict(architecture='gfx1201', head_dim=64, storage='f16',
                  compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                  rows=rows, evidence='static_exact_image', promotion_eligible=False,
                  missing=['runtime kernel attribution', 'independent clocks', 'native-Linux measurement'])
    (directory/'comparison.json').write_text(json.dumps(packet, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    record(parser.parse_args().directory)
