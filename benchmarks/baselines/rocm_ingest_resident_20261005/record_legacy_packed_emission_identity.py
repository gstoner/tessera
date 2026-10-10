"""Bind legacy packed HIP emissions to the sealed relabel-era generator.

Run in host WSL from the repository root. No timing packet or shader build is
rewritten; this checks source identity for every recorded packed flag set.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from tessera.compiler.rocm_mxfp4_packed_folded import emit_mxfp4_packed_folded_prefill_hip

ROOT=Path(__file__).resolve().parents[3]
PACKET=Path(__file__).resolve().parent
GENERATOR="python/tessera/compiler/rocm_mxfp4_packed_folded.py"
REVISION="993c009d7"
FLAGS={
    "table":dict(integer_decode=False),
    "integer":dict(integer_decode=True),
    "batched_b":dict(integer_decode=True,batched_loads=True),
    "batched_a":dict(integer_decode=True,batched_a_loads=True),
    "batched_ab":dict(integer_decode=True,batched_loads=True,batched_a_loads=True),
    "batched_b_pair_scale":dict(integer_decode=True,batched_loads=True,reuse_pair_scales=True),
    "permute":dict(integer_decode=False,batched_loads=True,permute_decode=True),
    "vector_pair":dict(integer_decode=False,permute_decode=True,vector_pair_loads=True),
    "a_base":dict(integer_decode=False,batched_loads=True,permute_decode=True,a_base_hoist=True),
    "a_offset32":dict(integer_decode=False,batched_loads=True,permute_decode=True,a_offset32=True),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def record():
    identity=json.loads((ROOT/"benchmarks/baselines/gfx1201_mxfp4_producer_relabel_20260924/identity.json").read_text())
    sealed=subprocess.run(["git","show",f"{REVISION}:{GENERATOR}"],cwd=ROOT,
        check=True,capture_output=True).stdout
    digest=sha(sealed)
    if digest!=identity["relabel_generator_sha256"][GENERATOR]:
        raise ValueError("historical source does not match the sealed relabel proof")
    name="tessera.compiler._sealed_packed_relabel_emission"
    module=ModuleType(name)
    module.__package__="tessera.compiler"
    module.__file__=str(ROOT/GENERATOR)
    sys.modules[name]=module
    try:
        exec(compile(sealed,module.__file__,"exec"),module.__dict__)
        emissions={}
        for label,flags in FLAGS.items():
            before=module.emit_mxfp4_packed_folded_prefill_hip(**flags)
            after=emit_mxfp4_packed_folded_prefill_hip(**flags)
            if before!=after:
                raise ValueError(f"legacy packed emission differs: {label}")
            emissions[label]=sha(before.encode())
    finally:
        sys.modules.pop(name,None)
    result={
        "schema":"tessera.rocm.packed_legacy_emission_identity.v1",
        "historical_revision":REVISION,"historical_generator_sha256":digest,
        "generator_path":GENERATOR,"inspected_current_file_sha256":sha((ROOT/GENERATOR).read_bytes()),
        "recorder_sha256":sha(Path(__file__).read_bytes()),
        "flags":FLAGS,"emission_sha256":emissions,
        "scope":"all ten recorded packed probe variants; emitted HIP source only, not fresh device timings",
    }
    (PACKET/"legacy_packed_emission_identity.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("legacy packed emitted-source identity passed for",len(emissions),"variants")

if __name__=="__main__":
    record()
