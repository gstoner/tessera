"""Fresh-process replay through the public product runtime, without compilers."""
import argparse
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from tessera.compiler.native_jvp import NativeJVPArtifact
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tessera.runtime import RuntimeArtifact, launch, backend_capabilities

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--artifact",type=Path,required=True)
    ap.add_argument("--digest",required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or "RTX 5070" not in device or device.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    # Device/runtime discovery may invoke system soname enumeration. Complete
    # it before forbidding every subprocess for package restoration/execution.
    backend_capabilities("nvidia_sm120")
    metadata=json.loads(args.artifact.read_text())
    artifact=NativeJVPArtifact(metadata["native_jvp"])
    artifact.validate()
    if artifact.artifact_hash!=args.digest:
        raise ValueError("public JVP differs from external pinned identity")
    def forbidden(*a,**k):
        raise AssertionError("compiler/process access during native product replay")
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        child=artifact.contract["steps"][0]["child_metadata"]
        program=NativeAttentionJVPProgram.from_json(child["program_json"],expected_digest=child["program_digest"])
        b,hq,hkv,sq,sk,d,dv=program.pair.forward.descriptor.provenance["shape"]
        policy=program.pair.forward.descriptor.provenance
        rng=np.random.default_rng(914)
        physical=[rng.normal(size=s).astype(np.float32)*.2 for s in
            ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))]
        tangents=[rng.normal(size=x.shape).astype(np.float32)*.1
            if i in program.active else np.zeros_like(x) for i,x in enumerate(physical)]
        def oracle(xs):
            q,k,v=(x.astype(np.float64) for x in xs)
            score=float(policy["scale"])*(q@np.swapaxes(np.repeat(k,hq//hkv,axis=1),-1,-2))
            if policy["causal"]:
                score=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),score,-np.inf)
            p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
            return p@np.repeat(v,hq//hkv,axis=1)
        h=1e-4
        expected=(oracle([x.astype(np.float64)+h*t for x,t in zip(physical,tangents,strict=True)])-
                  oracle([x.astype(np.float64)-h*t for x,t in zip(physical,tangents,strict=True)]))/(2*h)
        frontend=[None]*3
        for i,index in enumerate(program.input_indices): frontend[index]=physical[i]
        values=tuple(frontend)+tuple(tangents[i] for i in program.active)
        result=launch(RuntimeArtifact(metadata=artifact.runtime_metadata()),values)
        if not result.get("ok") or result.get("execution_mode")!="cuda_runtime":
            raise RuntimeError(f"public native product replay failed: {result}")
        primal,tangent=result["output"]
        np.testing.assert_allclose(primal,oracle(physical),atol=3e-5,rtol=3e-5)
        np.testing.assert_allclose(tangent,expected,atol=3e-5,rtol=3e-5)
    args.output.write_text(json.dumps(dict(device=device,artifact_hash=args.digest,
        correctness="passed",compiler_subprocesses="forbidden",
        max_abs_error=float(np.max(np.abs(tangent-expected)))),indent=2)+"\n")
    print("public product compiler-free replay passed",args.artifact,flush=True)
if __name__=="__main__": main()
