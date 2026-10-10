"""Exact SM120 broadcast saved-state packages and public JIT reverse proof."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import run_case as checked
from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import run_case as public
from tests._support.nvidia import nvidia_cuda_host_ready


def main():
    if not nvidia_cuda_host_ready():
        raise RuntimeError("owning SM120 host required")
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or gpu.split(",")[-1].strip() != "12.0":
        raise RuntimeError("requires one exact SM120 GPU")
    dims = (2,4,2,5,7,4,3)
    physical = [(1,4,5,7),(2,1,5,7),(2,4,1,7),(2,4,5,1),(1,1,1,7),(1,1,1,1)]
    rows = []
    for shape in physical:
        for causal in (False,True):
            rows.append(dict(
                checked=checked(dims,causal,samples=3,reps=100,bias_shape=shape),
                public=public(dims,causal,("bias","q","k","v"),samples=3,bias_shape=shape)))
    # Sq > Sk exercises the opposite end-aligned causal boundary.
    dims = (2,4,2,7,5,4,3)
    for causal in (False,True):
        shape=(1,1,7,1)
        rows.append(dict(checked=checked(dims,causal,samples=3,reps=100,bias_shape=shape),
                         public=public(dims,causal,("bias","v","q"),samples=3,bias_shape=shape)))
    without_bias_cotangent = [
        public((2,4,2,5,7,4,3),causal,("q","v"),samples=3,bias_shape=(1,1,1,7))
        for causal in (False,True)]
    explicit_qkv_only = [
        checked((2,4,2,5,7,4,3),causal,samples=3,reps=100,bias_shape=(1,1,1,7),bias_gradient=False)
        for causal in (False,True)]
    sources = ["python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/resident_attention.py","python/tessera/runtime.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "benchmarks/nvidia/benchmark_checkpoint_bias_gradient.py",
        "benchmarks/nvidia/benchmark_jit_attention_bias_vjp.py",
        "benchmarks/nvidia/benchmark_checkpoint_broadcast_package.py"]
    packet=dict(schema="tessera.nvidia.broadcast_checkpoint_package.v1",gpu=gpu,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        bridge_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        rows=rows,without_bias_cotangent=without_bias_cotangent,explicit_qkv_only=explicit_qkv_only,timing_scope="checked host staging wall time, resident backward dispatch events, public capture/backward wall time remain separate; no speedup claim")
    dest=Path(os.environ["TESSERA_BROADCAST_PACKAGE_PACKET"])
    dest.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(rows),"checked and public JIT broadcast checkpoint cases")


if __name__ == "__main__":
    main()
