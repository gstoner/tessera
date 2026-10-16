from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.record_gfx1201_folded_native_package import compile_for_tool
from benchmarks.rocm.benchmark_folded_input_pitches import source_state
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape,compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
root=Path("/home/angstorms/scratch/jvp-schema-reconcile-20261011")
old=root/"benchmarks/baselines/gfx1201_folded_cu_policy_20261016"
out=Path("/home/angstorms/scratch/gfx1201-m256-attribution-20261015/input-pitches")
tool=root/".build-schema-make/tools/tessera-opt/tessera-opt"
os.environ["TESSERA_OPT"]=str(tool)
assert rt._rocm_live_arch()=="gfx1201"
before=source_state(tool)
packet=json.loads(gzip.decompress((old/"cu-grid-forward.json.gz").read_bytes()))
rows=[]
for row in packet["rows"]:
 m,n,k=row["shape_mnk"];inputs=base._logical_inputs(base.Case("prefill",m,n,k))
 folded=prepare_folded_weights(inputs["packed_row_major"],inputs["b_scale"],allow_approximate=True)
 p=compile_for_tool(tool,inputs["a"],inputs["a_scale"],folded)
 expected=row["frontend_receipt"]
 actual=p.route_receipt
 fields=("hsaco_sha256","entry_symbol","abi_id","tile_ir_sha256","target_ir_sha256")
 assert all(actual[key]==expected[key] for key in fields),(m,n,k,{key:(actual[key],expected[key]) for key in fields})
 rows.append({"shape_mnk":[m,n,k],"current":{key:actual[key] for key in fields},
              "frozen":{key:expected[key] for key in fields},"identical":True})
 print("CU image parity",m,n,k,flush=True)
formats=[]
reference=json.loads((old/"format-image-equality.json").read_text())
for row in reference["rows"]:
 spec=BlockScaleShape(*row["shape_mnk"],32,1,row["layout"],"bf16")
 compile=compile_blockscale if row["format"]=="fp8_fp32_scales" else compile_mxfp8
 p=compile(spec)
 actual={"payload_sha256":hashlib.sha256(p.image.payload).hexdigest(),
         "abi":p.descriptor.abi_id,"entry":p.descriptor.entry_symbol,
         "tile_sha256":hashlib.sha256(p.tile_ir.encode()).hexdigest(),
         "target_sha256":hashlib.sha256(p.target_ir.encode()).hexdigest()}
 assert actual==row["arms"]["candidate"],(spec,row["format"],actual,row["arms"]["candidate"])
 formats.append({"shape_mnk":row["shape_mnk"],"layout":row["layout"],"format":row["format"],
                 "current":actual,"frozen":row["arms"]["candidate"],"identical":True})
 print("format image parity",row["shape_mnk"],row["layout"],row["format"],flush=True)
from tessera.compiler.rocm_mxfp4_folded_frontend import author_folded_scaled_matmul_graph, _lower
from tessera.compiler.rocm_mxfp4_folded_carrier import package_folded_scaled_wmma_target_ir
pitch_rows=[]
expected_image="f80b91072c88878e2be6193b83e5ca72a6d59b033da0859a3b05ad31aba1b1fc"
for n in (8192,16384):
 for k in (4032,4096,4160,5120,8192):
  inputs=base._logical_inputs(base.Case("prefill",256,n,k))
  folded=prepare_folded_weights(inputs["packed_row_major"],inputs["b_scale"],allow_approximate=True)
  for lda,ldb in ((k,k),(k+16,k),(k,k+16),(k+16,k+16)):
   for canonicalize in (False,True):
    if canonicalize:
     graph=author_folded_scaled_matmul_graph(256,n,k,input_row_strides=(lda,ldb))
     graph=subprocess.run([str(tool),"--canonicalize"],input=graph,text=True,capture_output=True,check=True).stdout
     p=package_folded_scaled_wmma_target_ir(_lower(tool,graph,target=False),
         _lower(tool,graph,target=True),folded,allow_approximate=True)
    else:
     p=compile_for_tool(tool,inputs["a"],inputs["a_scale"],folded,input_row_strides=(lda,ldb)).package
    assert p.image.payload_digest==expected_image,(n,k,lda,ldb,canonicalize,p.image.payload_digest)
    pitch_rows.append({"shape_mnk":[256,n,k],"pitches":[lda,ldb],
                       "canonicalized":canonicalize,"payload_sha256":p.image.payload_digest,
                       "entry":p.descriptor.entry_symbol,"abi":p.descriptor.abi_id})
  print("pitched image parity",n,k,flush=True)
after=source_state(tool);assert before==after
record={"schema":"tessera.rocm.folded_input_pitch_legacy_parity.v1",
        "architecture":rt._rocm_live_arch(),"source_before":before,"source_after":after,
        "scope":"exact native Tile, Target, ABI, entry and HSACO equality; no new format performance claim",
        "frozen_cu_revision":"68fb444434bbb20f396399e8a1df186c67e00f94",
        "frozen_format_revision":reference["source_commit"],"cu_rows":rows,"format_rows":formats,"pitched_rows":pitch_rows,
        "measured_source_revision":"920bdadc4212fb89a22bb65dc05cffd169af6bf1"}
(out/"post-canonicalization-image-parity.json").write_text(json.dumps(record,indent=2)+"\n")
print("all legacy image controls preserved",len(rows),len(formats),flush=True)
