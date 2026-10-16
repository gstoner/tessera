from pathlib import Path
import hashlib
import json
import os
import subprocess
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.record_gfx1201_folded_native_package import compile_for_tool
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera import runtime as rt

out=Path(__file__).resolve().parent
current=Path(os.environ["TESSERA_OPT"]).resolve()
control=out.parent/"native-control/tessera-opt"
expected="8e53d33556d04066c4a6cd0335b7be1b164b1f80"
def state():
    return {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "status":subprocess.check_output(["git","status","--porcelain"],text=True).strip(),
            "compiler_sha256":hashlib.sha256(current.read_bytes()).hexdigest()}
before=state()
assert before["head"]==expected and not before["status"]
assert rt._rocm_live_arch()=="gfx1201"
rows=[]
for m,n,k in [(256,4096,1536),(256,17408,1536),(256,8192,1024),
               (256,16384,2048),(256,8192,5120),(256,16384,5120),
               (200,8192,1536),(512,8192,1536),(256,36864,1536)]:
    case=base.Case("prefill",m,n,k)
    inputs=base._logical_inputs(case)
    folded=prepare_folded_weights(inputs["packed_row_major"],inputs["b_scale"],allow_approximate=True)
    arms={}
    for name,tool in (("current",current),("control",control)):
        p=compile_for_tool(tool,inputs["a"],inputs["a_scale"],folded,runtime_k=True)
        arms[name]={"image_sha256":hashlib.sha256(p.package.image.payload).hexdigest(),
                    "entry":p.package.descriptor.entry_symbol,"abi":p.package.descriptor.abi_id,
                    "target_sha256":hashlib.sha256(p.package.target_ir.encode()).hexdigest(),
                    "tile_sha256":hashlib.sha256(p.package.tile_ir.encode()).hexdigest()}
    assert arms["current"]==arms["control"],(m,n,k,arms)
    rows.append({"shape_mnk":[m,n,k],"identical":True,"arms":arms})
    print(m,n,k,"byte-identical",flush=True)
after=state()
assert before==after
record={"source_before":before,"source_after":after,"architecture":"gfx1201",
        "control_sha256":hashlib.sha256(control.read_bytes()).hexdigest(),
        "scope":"compiled excluded classes; no device timing or sibling claim","rows":rows}
(out/"excluded-image-equality.json").write_text(json.dumps(record,indent=2)+"\n")
