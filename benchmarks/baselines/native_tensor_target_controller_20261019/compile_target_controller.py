import base64, gzip, hashlib, json, re, subprocess
from pathlib import Path
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.nvidia_native import _compile_tile_ir
p=Path("/home/angstorms/scratch/w11-program-pipeline-20261019")
out=p/"target-candidate";out.mkdir(exist_ok=True)
records=[]
def digest(b): return hashlib.sha256(b).hexdigest()
for graph in sorted((Path.cwd()/"benchmarks/baselines/native_tensor_program_pipeline_20261019").glob("*.graph.mlir")):
 source=graph.read_text()
 full=subprocess.run([str(find_tessera_opt()),"-","--tessera-nvidia-pipeline-sm120",
                     "--tessera-lower-to-nvidia-sm120"],
                     input=source,text=True,capture_output=True,check=True).stdout
 name=graph.name.removesuffix(".graph.mlir")
 assert not re.search(r"^\s*(?:%\w+ = )?tile\.",full,re.M)
 frame=json.loads(base64.b64decode(re.search(r'tessera.native.tensor_program_frame_json = "([^"]+)"',full)[1]))
 assert full.count("tessera_nvidia.program_invoke ") == len(frame["stages"])
 (out/(name+".target.mlir.gz")).write_bytes(gzip.compress(full.encode(),mtime=0))
 old=json.loads((p/"candidate"/(name+".frame.json")).read_text())
 for index,stage in enumerate(frame["stages"]):
  member=subprocess.run([str(find_tessera_opt()),"-",f"--tessera-schedule-to-tile=tensor-member={index}"],
                        input=full,text=True,capture_output=True,check=True).stdout
  assert "program_invoke " not in member
  _,ptx,*_=_compile_tile_ir(member,stage["entry"])
  file=digest(ptx.encode())+".ptx"
  (out/file).write_text(ptx)
  stage["ptx_file"]=file;stage["ptx_sha256"]=digest(ptx.encode())
  assert stage == old["stages"][index], (name,index,stage,old["stages"][index])
 frame["source_sha256"]=digest(source.encode())
 frame["program_sha256"]=digest(full.encode())
 assert frame["source_sha256"] == old["source_sha256"]
 (out/(name+".frame.json")).write_text(json.dumps(frame,indent=2)+"\n")
 print(name,len(frame["stages"]),"Target stages compiled; identical device images",flush=True)
 records.append({"name":name,"frame":name+".frame.json","stages":len(frame["stages"]),
                 "source_sha256":frame["source_sha256"],"target_program_sha256":frame["program_sha256"],
                 "device_images_identical_to_tile_control":True})
(out/"index.json").write_text(json.dumps({"compiler_sha256":digest(find_tessera_opt().read_bytes()),"cases":records},indent=2)+"\n")
