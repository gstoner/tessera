import hashlib
import json
import os
from pathlib import Path
from tessera import runtime as rt
root=Path('/home/angstorms/scratch/rocm-native-gate-up-views-20261016/k16-quant')
out=root/'public-package-evidence';out.mkdir(exist_ok=True)
for m,k in [(3,32),(19,12288),(256,12288)]:
 artifact=rt.RuntimeArtifact.from_json((root/f'public-quant-{m}-{k}.json').read_text())
 image=artifact.native_image;member=json.loads(artifact.launch_descriptor.provenance['native_block_quant']['member_json'])
 prefix=out/f'quant-{m}-{k}';prefix.with_suffix('.hsaco').write_bytes(image.payload)
 prefix.with_suffix('.json').write_text(json.dumps({'shape':[m,k],'member':member,
  'image_sha256':hashlib.sha256(image.payload).hexdigest(),'compiler_sha256':hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
  'artifact_sha256':hashlib.sha256((root/f'public-quant-{m}-{k}.json').read_bytes()).hexdigest()},indent=2)+'\n')
