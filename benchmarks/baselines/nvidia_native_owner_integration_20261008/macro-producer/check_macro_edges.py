import importlib.util,sys,hashlib,json
from pathlib import Path
import numpy as np
import tessera
name="tessera.compiler.prepared_nvidia_lhs"
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location(name,root/"prepared_nvidia_lhs.py")
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,_storage,_oracle
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
plain={"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs}
fused={"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
rows=[]
rng=np.random.default_rng(20261008)
for dtype in ("fp16","bf16"):
 for kind in plain:
  for epilogue in (False,True):
   source=(rng.normal(size=(128,4096))*.2).astype(_storage(dtype))
   rhs=np.array(rng.normal(size=(4096,64))*.2,dtype=_storage(dtype),order="F")
   bias=(rng.normal(size=64)*.02).astype(np.float32)
   residual=(rng.normal(size=(128,64))*.02).astype(np.float32)
   operands=(source,rhs,bias,residual) if epilogue else (source,rhs)
   fn=(fused if epilogue else plain)[kind]
   program=fn.compile_native_lhs_matmul(*operands)
   descriptor=program.edge.consumer.descriptor
   expected_route=("typed_fragment_global" if epilogue else "macro_cta_cp_async_2stage_shared_ab_"+("f16" if dtype=="fp16" else "bf16"))
   assert descriptor.provenance["physical_route"]==expected_route
   assert descriptor.geometry.policy==("sm120_scheduled_typed_16x8_mn" if epilogue else "sm120_scheduled_macro_cta_32x32_mn")
   call=PreparedLhsCall(program)
   try:
    first,receipt=call(operands)
    expected=_oracle(source,rhs,kind,bias if epilogue else None,residual if epilogue else None)
    np.testing.assert_allclose(first,expected,rtol=.015,atol=.015)
    saved=first.copy()
    changed=tuple([-source,*operands[1:]])
    second,receipt2=call(changed)
    expected2=_oracle(changed[0],rhs,kind,bias if epilogue else None,residual if epilogue else None)
    np.testing.assert_allclose(second,expected2,rtol=.015,atol=.015)
    np.testing.assert_array_equal(first,saved)
    assert receipt["execution_kind"]==receipt2["execution_kind"]=="native_gpu"
    rows.append(dict(dtype=dtype,producer=kind,epilogue=epilogue,shape_mkn=[128,4096,64],physical_route=expected_route,entry=descriptor.entry_symbol,image_digest=program.edge.consumer.image.image_digest,descriptor_digest=descriptor.descriptor_digest,max_abs_error=float(np.max(np.abs(first.astype(np.float64)-expected))),changed_max_abs_error=float(np.max(np.abs(second.astype(np.float64)-expected2))),retained_result="bit_identical",correctness="passed"))
   finally:call.close()
packet=dict(device="NVIDIA GeForce RTX 5070",device_uuid="GPU-cba12639-821a-7a10-4cd3-f918f9c0a545",architecture="sm_120",runtime_sha256=hashlib.sha256((root/"libtessera_nvidia_ptx_launch.so").read_bytes()).hexdigest(),native_source_sha256=hashlib.sha256((root/"matmul_prepared.cpp").read_bytes()).hexdigest(),python_source_sha256=hashlib.sha256((root/"prepared_nvidia_lhs.py").read_bytes()).hexdigest(),rows=rows,timing="not_measured_during_aggregate_validation")
(root/"macro_edges.json").write_text(json.dumps(packet,indent=2)+"\n")
print(json.dumps(dict(rows=len(rows),max_abs_error=max(r["max_abs_error"] for r in rows))))
