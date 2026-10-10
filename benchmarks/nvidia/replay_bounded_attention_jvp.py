"""Fresh-process, compiler-free replay of the named bounded SM120 JVP packet."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from benchmarks.record_device_ring_protocol import Device
from benchmarks.nvidia.benchmark_bounded_attention_jvp import execute
from tessera.compiler.attention_shape_contract import DYNAMIC_DIM
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    os.environ["TESSERA_OPT"] = "/missing/compiler"
    os.environ["TESSERA_NVIDIA_OPT"] = "/missing/target-compiler"

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler/process access during bounded replay")

    with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden), patch("subprocess.check_output", forbidden):
        program = NativeAttentionJVPProgram.from_json(args.program.read_text(), expected_digest=args.digest)
        policy = program.pair.forward.descriptor.provenance
        if (tuple(policy["shape"]) != (1, 2, 1, DYNAMIC_DIM, DYNAMIC_DIM, 4, 3)
                or tuple(policy.get("shape_bounds", ())) != (1, 2, 1, 9, 11, 4, 3)):
            raise ValueError("replay requires the recorded bounded JVP envelope")
        bias_shape = tuple(policy.get("bias_shape", ()))
        bias = ("broadcast" if bias_shape[-2:] == (1, 1) else "full") if policy.get("bias") else None
        device = Device("nvidia")
        rows = [execute(device, program, sq, sk, policy["causal"], bias)
                for sq, sk in ((1, 1), (7, 3), (9, 11))]
    args.output.write_text(json.dumps(dict(device=gpu, program_digest=args.digest,
        compiler_subprocesses="forbidden", exact_device_replay="passed", rows=rows), indent=2)+"\n")
    print("compiler-free bounded replay passed", flush=True)


if __name__ == "__main__":
    main()
