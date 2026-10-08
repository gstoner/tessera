from pathlib import Path
import importlib.util,sys,os,json,hashlib,subprocess
root=Path(__file__).parent
tool=root/"tessera-opt-contract"
os.environ["TESSERA_OPT"]=str(tool)
name="tessera.compiler.scheduled_kernel"
spec=importlib.util.spec_from_file_location(name,root/"scheduled_kernel.py")
scheduled=importlib.util.module_from_spec(spec);sys.modules[name]=scheduled;spec.loader.exec_module(scheduled)
from tests.unit.test_scheduled_kernel_consumers import _module
from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_matmul import run_tessera_opt
control_name="tessera.compiler.serial_softmax_control"
control_spec=importlib.util.spec_from_file_location(control_name,Path.cwd()/"python/tessera/compiler/scheduled_kernel.py")
control=importlib.util.module_from_spec(control_spec);sys.modules[control_name]=control;control_spec.loader.exec_module(control)
primary=Path.cwd()/".build-sm120-w1-1/tools/tessera-opt/tessera-opt"
rows=[]
for dtype in ("fp16","bf16","fp32"):
 module=_module(family="softmax",target="nvidia_sm120")
 fn=module.functions[0];storage=tensor_ir_type((2,3,4096),dtype)
 fn.args[0].ir_type=storage;fn.result_types=[storage]
 op=fn.body[0];op.operand_types=[str(storage)];op.result_type=str(storage);op.inferred_type=storage
 os.environ["TESSERA_OPT"]=str(primary)
 baseline=control.lower_scheduled_kernel(module,target="nvidia_sm120")
 os.environ["TESSERA_OPT"]=str(tool)
 default=scheduled.lower_scheduled_kernel(module,target="nvidia_sm120")
 assert baseline.graph_ir==default.graph_ir
 assert baseline.schedule_ir==default.schedule_ir
 assert baseline.tile_ir==default.tile_ir
 serial=scheduled.lower_scheduled_kernel(module,target="nvidia_sm120",schedule="serial")
 coop=scheduled.lower_scheduled_kernel(module,target="nvidia_sm120",schedule="cooperative_128")
 assert coop.schedule=="cooperative_128"
 assert serial.schedule_digest!=coop.schedule_digest
 assert 'schedule = "cooperative_128"' in coop.schedule_ir
 assert 'schedule = "cooperative_128"' in coop.tile_ir
 assert '_cooperative_128' in coop.tile_ir
 assert run_tessera_opt(tool,coop.schedule_ir,"--tessera-schedule-to-tile")==coop.tile_ir
 lowered=run_tessera_opt(tool,coop.tile_ir,"--tessera-lower-to-nvidia-sm120")
 assert lowered.count("nvvm.barrier")==18
 assert "tile.softmax_kernel" not in lowered
 (root/(dtype+"-graph.mlir")).write_text(coop.graph_ir)
 (root/(dtype+"-schedule.mlir")).write_text(coop.schedule_ir)
 (root/(dtype+"-tile.mlir")).write_text(coop.tile_ir)
 (root/(dtype+"-target.mlir")).write_text(lowered)
 # Alter only the physical Schedule record, leaving its semantic Graph owner intact.
 lines=coop.schedule_ir.splitlines(True)
 forged="".join(line.replace('schedule = "cooperative_128"','schedule = "serial"') if "schedule.softmax" in line else line for line in lines)
 assert forged!=coop.schedule_ir
 r=subprocess.run([str(tool),"-","--tessera-schedule-to-tile"],input=forged,text=True,capture_output=True)
 assert r.returncode!=0 and "altered after hashing" in r.stderr,r.stderr
 # Tile cannot claim cooperative geometry without the checked workgroup.
 forged_tile=coop.tile_ir.replace("tessera.workgroup_size = 128 : i64","tessera.workgroup_size = 32 : i64")
 assert forged_tile!=coop.tile_ir
 r=subprocess.run([str(tool),"-","--tessera-lower-to-nvidia-sm120"],input=forged_tile,text=True,capture_output=True)
 assert r.returncode!=0 and "workgroup size 128" in r.stderr,r.stderr
 rows.append(dict(dtype=dtype,default_serial_boundaries="byte_identical_to_primary",serial_schedule_digest=serial.schedule_digest,cooperative_schedule_digest=coop.schedule_digest,schedule_replay="byte_identical",forged_policy="rejected",forged_geometry="rejected",native_barriers=18))
for target in ("x86","apple_gpu","rocm_gfx1151","rocm_gfx1201"):
 try:scheduled.lower_scheduled_kernel(_module(family="softmax",target=target),target=target,schedule="cooperative_128")
 except ValueError:pass
 else:raise AssertionError("sibling admitted SM120 physical schedule: "+target)
packet=dict(compiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),rows=rows,sibling_policy="rejected_on_all_four_targets",execution_evidence="pending",native_launch_contract="pending")
(root/"schedule_contract_proof.json").write_text(json.dumps(packet,indent=2)+"\n")
print(json.dumps(packet))
