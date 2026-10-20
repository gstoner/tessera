import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.nvidia_native import _compile_tile_ir
p=Path("/home/angstorms/scratch/w11-program-pipeline-20261019")
out=p/"candidate";out.mkdir(exist_ok=True)
records=[]
for graph in sorted(p.glob("*.graph.mlir")):
 source=graph.read_text()
 full=subprocess.run([str(find_tessera_opt()),"-","--tessera-nvidia-pipeline-sm120"],
                     input=source,text=True,capture_output=True,check=True).stdout
 frame=json.loads(base64.b64decode(re.search(r'tessera.native.tensor_program_frame_json = "([^"]+)"',full)[1]))
 name=graph.name.removesuffix(".graph.mlir")
 (out/(name+".program.mlir")).write_text(full)
 for index,stage in enumerate(frame["stages"]):
  member=subprocess.run([str(find_tessera_opt()),"-",f"--tessera-schedule-to-tile=tensor-member={index}"],
                        input=full,text=True,capture_output=True,check=True).stdout
  (out/f"{name}.stage{index}.tile.mlir").write_text(member)
  _,ptx,*_=_compile_tile_ir(member,stage["entry"])
  file=hashlib.sha256(ptx.encode()).hexdigest()+".ptx"
  (out/file).write_text(ptx)
  stage["ptx_file"]=file;stage["ptx_sha256"]=hashlib.sha256(ptx.encode()).hexdigest()
 frame["source_sha256"]=hashlib.sha256(source.encode()).hexdigest()
 frame["program_sha256"]=hashlib.sha256(full.encode()).hexdigest()
 (out/(name+".frame.json")).write_text(json.dumps(frame,indent=2)+"\n")
 print(name, len(frame["stages"]), "native stages compiled",flush=True)
 records.append({"name":name,"frame":name+".frame.json","stages":len(frame["stages"])})
(out/"index.json").write_text(json.dumps({"compiler_sha256":hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),"cases":records},indent=2)+"\n")
