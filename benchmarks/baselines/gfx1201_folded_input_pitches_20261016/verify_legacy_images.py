from pathlib import Path
import gzip,hashlib,json,os
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
after=source_state(tool);assert before==after
record={"schema":"tessera.rocm.folded_input_pitch_legacy_parity.v1",
        "architecture":rt._rocm_live_arch(),"source_before":before,"source_after":after,
        "scope":"exact native Tile, Target, ABI, entry and HSACO equality; no new format performance claim",
        "frozen_cu_revision":"68fb444434bbb20f396399e8a1df186c67e00f94",
        "frozen_format_revision":reference["source_commit"],"cu_rows":rows,"format_rows":formats}
(out/"legacy-image-parity.json").write_text(json.dumps(record,indent=2)+"\n")
print("all legacy image controls preserved",len(rows),len(formats),flush=True)
