from pathlib import Path
import hashlib
import json
import os
import subprocess
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape,compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
root=Path("/home/angstorms/scratch/jvp-schema-reconcile-20261011")
out=Path("/home/angstorms/scratch/gfx1201-m256-attribution-20261015/output-slab-design/frozen-cu-policy")
tools={"baseline":Path("/home/angstorms/scratch/gfx1201-m256-attribution-20261015/fresh-reference/tessera-opt"),"candidate":root/".build-schema-make/tools/tessera-opt/tessera-opt"}
if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
rows=[]
for shape in [(17,19,64),(200,256,128),(256,512,1024)]:
 for layout in ["kn","nk"]:
  spec=BlockScaleShape(*shape,32,1,layout,"bf16")
  for name,compile in [("fp8_fp32_scales",compile_blockscale),("mxfp8_e8m0_scales",compile_mxfp8)]:
   arms={}
   for label,tool in tools.items():
    os.environ["TESSERA_OPT"]=str(tool)
    p=compile(spec)
    arms[label]={"payload_sha256":hashlib.sha256(p.image.payload).hexdigest(),"abi":p.descriptor.abi_id,"entry":p.descriptor.entry_symbol,"tile_sha256":hashlib.sha256(p.tile_ir.encode()).hexdigest(),"target_sha256":hashlib.sha256(p.target_ir.encode()).hexdigest()}
   assert arms["baseline"]==arms["candidate"],(shape,layout,name,arms)
   rows.append({"shape_mnk":shape,"layout":layout,"format":name,"identical_native_package":True,"arms":arms})
record={"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"architecture":rt._rocm_live_arch(),"source_clean":not bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()),"scope":"K32 matched FP8/MXFP8 controls against immutable baseline; frozen native CU profile is folded-only; no new format performance claim","tools":{name:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in tools.items()},"candidate_cpp_override_sha256":hashlib.sha256((root/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp").read_bytes()).hexdigest(),"rows":rows}
(out/"format-controls/image-equality.json").write_text(json.dumps(record,indent=2)+"\n")
print("Identical compiled Tile/Target/image/ABI controls:",len(rows),flush=True)
