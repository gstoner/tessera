"""Fresh-process replay of a pinned saved-LSE reverse runtime product."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from tessera.runtime import RuntimeArtifact, backend_capabilities, launch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--digest",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or "RTX 5070" not in device or device.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    backend_capabilities("nvidia_sm120")
    def forbidden(*a,**k):
        raise AssertionError("compiler/process access during native product replay")
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        artifact=RuntimeArtifact.from_json(args.artifact.read_text())
        if artifact.artifact_hash!=args.digest:
            raise ValueError("reverse artifact differs from external pinned identity")
        with np.load(args.artifact.with_suffix(".npz")) as data:
            values=tuple(data[name] for name in artifact.metadata["arg_names"])
            expected=tuple(data[f"expected_{i}"] for i in range(len([n for n in data.files if n.startswith("expected_")])))
        receipt=launch(artifact,values)
        if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
            raise RuntimeError(f"reverse replay failed: {receipt}")
        gradients=tuple(receipt["output"])
        assert len(gradients)==len(expected)
        for x,reference in zip(gradients,expected,strict=True):
            np.testing.assert_allclose(x,reference,atol=3e-5,rtol=3e-5)
    args.output.write_text(json.dumps(dict(device=device,artifact_hash=args.digest,
        correctness="passed",compiler_subprocesses="forbidden",
        physical_attestation=receipt.get("physical_attestation"),
        max_abs_error=max(float(np.max(np.abs(x-y))) for x,y in zip(gradients,expected,strict=True))),indent=2)+"\n")
    print("public reverse compiler-free replay passed",args.artifact,flush=True)
if __name__=="__main__":main()
