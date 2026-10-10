"""Retain numerical failures while comparing frozen/current native norm compilers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np
from tessera import runtime as rt
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs,runtime_artifact
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs,layer_lhs,_storage,_oracle
from tests.device.nvidia.test_cooperative_norm import package,launch,oracle

def record():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("owning SM120 required")
    current=os.environ["TESSERA_NVIDIA_OPT"]
    baseline=str(Path(".build-sm120-w1-1/norm-accuracy-ab-20261005/tessera-nvidia-opt-baseline").resolve())
    rows=[]
    try:
        for dtype in ("bf16","fp16"):
            for kind,fn in (("rmsnorm",rms_lhs),("layernorm",layer_lhs)):
                for k in (1024,4096,8192):
                    for seed in (120517,512):
                        rng=np.random.default_rng(seed)
                        source=(rng.normal(size=(128,k))*.2).astype(_storage(dtype))
                        rhs=(rng.normal(size=(k,64))*.2).astype(_storage(dtype))
                        expected=_oracle(source,rhs,kind)
                        ideal=oracle(source,kind)
                        for arm,tool in (("baseline",baseline),("candidate",current)):
                            os.environ["TESSERA_NVIDIA_OPT"]=tool
                            for mode in ("serial","cooperative_128"):
                                _,p=package(kind,source,mode)
                                y=launch(p,source)
                                program=package_traced_lhs(fn._traced_autodiff_module((source,rhs),{}),producer_schedule=mode)
                                result=rt.launch(runtime_artifact(program),(source,rhs))
                                if not result.get("ok"):raise RuntimeError(result)
                                output=result["output"]
                                consumer_expected=y.astype(np.float64)@rhs.astype(np.float64)
                                err=np.abs(output-expected)
                                rows.append(dict(dtype=dtype,kind=kind,shape_mkn=[128,k,64],seed=seed,
                                    arm=arm,schedule=mode,
                                    original_tolerance_violations=int(np.count_nonzero(err>.015+.015*np.abs(expected))),
                                    output_max_abs_error=float(np.max(err)),
                                    norm_different_elements=int(np.count_nonzero(y!=ideal)),
                                    norm_max_abs_error=float(np.max(np.abs(y.astype(np.float64)-ideal.astype(np.float64)))),
                                    consumer_max_abs_error=float(np.max(np.abs(output-consumer_expected))),
                                    producer_image=p.image.image_digest,
                                    consumer_image=program.edge.consumer.image.image_digest,
                                    consumer_payload=program.edge.consumer.image.payload_digest,
                                    consumer_target_ir=program.edge.consumer.image.target_ir_digest))
                        print(dtype,kind,k,seed,flush=True)
    finally:
        os.environ["TESSERA_NVIDIA_OPT"]=current
    return dict(schema="tessera.nvidia.norm_accuracy.v1",gpu=gpu,rows=rows,
        compiler_sha256={arm:hashlib.sha256(Path(p).read_bytes()).hexdigest() for arm,p in (("baseline",baseline),("candidate",current))},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Numerical attribution only; failures remain visible, no performance or completion claim.",
        oracle="Independent float64 norm rounded to declared intermediate storage, then float64 matmul. Original rtol/atol=.015.")
if __name__=="__main__":
    Path(os.environ["TESSERA_NORM_ACCURACY_PACKET"]).write_text(json.dumps(record(),indent=2,sort_keys=True)+"\n")
