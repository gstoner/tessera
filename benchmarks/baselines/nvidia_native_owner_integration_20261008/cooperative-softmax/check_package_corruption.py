from pathlib import Path
import dataclasses
root=Path(__file__).parent
exec((root/"check_package_device.py").read_text().split("rows=[]")[0])
from tessera.compiler.native_artifact import LaunchGeometry
x=np.ones((3,17),np.float32)
graph=ordinary._traced_autodiff_module((x,),{})
artifact=scheduled.lower_scheduled_kernel(graph,target="nvidia_sm120",schedule="cooperative_128")
p=nvidia_native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
bindings=sorted(p.descriptor.buffers,key=lambda b:b.ordinal)
buffers={bindings[0].name:x,bindings[1].name:np.empty_like(x)}
original=rt._register_nvidia_ptx
def forbidden(*a,**k):raise AssertionError("forged descriptor reached CUDA registration")
rt._register_nvidia_ptx=forbidden
checks=[]
try:
 for label,changes in (("serial_geometry",dict(geometry=LaunchGeometry(policy="sm120_softmax_thread_per_row_128"))),("wrong_workgroup",dict(geometry=LaunchGeometry(grid=(3,1,1),workgroup=(32,1,1)))),("serial_policy",dict(provenance={**p.descriptor.provenance,"schedule":"serial"})),("wrong_storage",dict(provenance={**p.descriptor.provenance,"storage":"f16"}))):
  desc=dataclasses.replace(p.descriptor,**changes)
  try:rt._submit_nvidia_sm120_native(p.image,desc,buffers,{"Rows":3,"K":17},None)
  except RuntimeError as e:
   assert "schedule/entry/geometry ABI mismatch" in str(e),str(e)
   checks.append(label)
  else:raise AssertionError(label)
finally:rt._register_nvidia_ptx=original
(root/"runtime_corruption_proof.json").write_text(json.dumps(dict(checks=checks,cuda_registration="not_reached"),indent=2)+"\n")
import pytest
raise SystemExit(pytest.main(["-c",str(Path.cwd()/"pyproject.toml"),"-q",str(root/"test_nvidia_scheduled_kernel_contract.py")]))

