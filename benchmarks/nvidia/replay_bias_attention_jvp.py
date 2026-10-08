"""Fresh-process portable common-runtime replay with compiler calls forbidden."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from tessera.runtime import RuntimeArtifact,launch,backend_capabilities
from tessera.compiler.native_attention_jvp_runtime import clear_prepared

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--artifact",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("owning RTX5070 / SM120 required")
    metadata=json.loads(args.artifact.read_text());artifact=RuntimeArtifact(metadata=metadata)
    inputs=args.artifact.with_suffix(".npz")
    with np.load(inputs) as data:
        values=tuple(data[f"arg_{i}"] for i in range(len(data.files)-2))
        expected=(data["expected_primal"].copy(),data["expected_tangent"].copy())
    backend_capabilities("nvidia_sm120")
    def forbidden(*a,**kw):raise AssertionError("compiler subprocess in portable replay")
    try:
        with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
            result=launch(artifact,values)
            assert result.get("ok") and result.get("execution_mode")=="cuda_runtime",result
            output=result["output"]
            for actual,oracle in zip(output,expected,strict=True):
                np.testing.assert_allclose(actual,oracle,atol=3e-5,rtol=3e-5)
        args.output.write_text(json.dumps(dict(device=gpu,architecture="sm_120",
            artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
            oracle_inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
            max_abs_error=float(np.max(np.abs(output[1]-expected[1]))),
            correctness="compiler_free_fresh_process_common_runtime"),indent=2)+"\n")
    finally:clear_prepared()
if __name__=="__main__":main()
