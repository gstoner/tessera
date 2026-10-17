"""Owning-host candidate image construction; not the final public packager."""
from pathlib import Path
import base64,hashlib,json,os,subprocess
from tests.unit.test_native_gate_up_view_schedule import source
from tessera.compiler.rocm_native import _extract_hsaco
from tessera.compiler.native_scaled_program import _manifest

v=Path('/home/angstorms/scratch/rocm-native-gate-up-views-20261016')
r=Path('/home/angstorms/scratch/sm120-relu-producer-20261014')
opt=Path(os.environ['TESSERA_OPT']);rocm=r/'.build-relu/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt'
rows=[]
for m,width,reverse in [(3,17,False),(3,17,True),(16,17,False),(16,64,False),(256,12288,False)]:
 stem=f'native-view-candidate-{m}-{width}-{reverse}'
 graph=source(m,width,reverse)
 def run(tool,text,pipeline):
  p=subprocess.run([str(tool),'--pass-pipeline='+pipeline],input=text,text=True,capture_output=True,timeout=180)
  (v/(stem+'-compiler-stderr.log')).open('a').write(p.stderr)
  if p.returncode:raise RuntimeError(p.stderr)
  return p.stdout
 schedule=run(opt,graph,'builtin.module(tessera-graph-to-schedule)')
 tile=run(opt,schedule,'builtin.module(tessera-schedule-to-tile)')
 target=run(opt,tile,'builtin.module(lower-tile-to-rocm{arch=gfx1201})')
 gpu=run(opt,target,'builtin.module(lower-tessera-target-to-rocdl)')
 # Only registered native MLIR lowering passes finish this diagnostic. The
 # production plugin/packager admission remains a separate required gate.
 binary=run(opt,gpu,'builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx1201},reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201 wave64=false},gpu-module-to-binary)')
 program=_manifest(binary,'tessera.rocm.parent_view_program_json')
 member=_manifest(binary,'tessera.rocm.parent_view_member_json')
 image=_extract_hsaco(binary)
 for name,text in [('graph',graph),('schedule',schedule),('tile',tile),('target',target),('gpu',gpu),('binary',binary)]:
  (v/(stem+'-'+name+'.mlir')).write_text(text)
 packet={'scope':'native candidate; final public JIT/plugin integration remains open','shape':[m,width],'reverse':reverse,'program_json':json.dumps(program,sort_keys=True),'members_json':[json.dumps(member,sort_keys=True)],'images':[base64.b64encode(image).decode()],'graph_ir':graph,'image_sha256':hashlib.sha256(image).hexdigest()}
 (v/(stem+'.json')).write_text(json.dumps(packet,indent=2)+'\n');(v/(stem+'.hsaco')).write_bytes(image)
 rows.append({'packet':stem+'.json','shape':[m,width],'reverse':reverse,'image_sha256':packet['image_sha256'],'entry':member['entry']})
 print(json.dumps(rows[-1]),flush=True)
(v/'native-view-candidate-index.json').write_text(json.dumps({'rows':rows,'scope':'native image construction only; no GPU execution claim'},indent=2)+'\n')
