import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tests.unit.test_native_k16_quant_target import stages,member
from tessera.compiler.rocm_native import _extract_hsaco,_serializer_env
root=Path('/home/angstorms/scratch/rocm-native-gate-up-views-20261016/k16-quant')
tool=shutil.which('mlir-opt')
assert tool
pipeline=('builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx1201},reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary)')
records=[]
for m,k in [(3,32),(19,12288),(256,12288)]:
 tile,target,gpu=stages(m,k)
 result=subprocess.run([tool,f'--pass-pipeline={pipeline}'],input=gpu,text=True,capture_output=True,env=_serializer_env())
 if result.returncode:raise RuntimeError(result.stderr)
 payload=_extract_hsaco(result.stdout)
 prefix=root/f'quant-{m}-{k}'
 prefix.with_suffix('.hsaco').write_bytes(payload)
 prefix.with_suffix('.gpu.mlir').write_text(gpu)
 manifest=member(target)
 record={'shape':[m,k],'member':manifest,'image_sha256':hashlib.sha256(payload).hexdigest(),
         'tessera_opt_sha256':hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
         'mlir_opt_sha256':hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
         'recorder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 prefix.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
 records.append(record)
 print('built',m,k,record['image_sha256'],flush=True)
assert len({r['image_sha256'] for r in records})==1
(root/'target-image-records.json').write_text(json.dumps(records,indent=2)+'\n')
