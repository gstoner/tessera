"""Compare actual cold packaging work; device execution is gated separately."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import subprocess
import time
from pathlib import Path
from tessera import runtime
from tessera.compiler import rocm_typed_scaled_native as candidate
from tessera.compiler.native_scaled_program import NativeScaledProgram
from tests.unit.test_native_typed_scaled_vmap import case

def run(control_path):
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 device required")
    name="tessera.compiler._single_image_control"
    spec=importlib.util.spec_from_file_location(name,control_path)
    control=importlib.util.module_from_spec(spec)
    sys.modules[name]=control
    spec.loader.exec_module(control)
    rows=[]
    for policy in (None,"shared_rhs_rows","independent_rhs","shared_lhs"):
        for fmt in ("fp32","e8m0"):
            for nk in (False,True):
                scalar,owner,values,_=case(policy or "independent_rhs",fmt,nk)
                if policy is None:
                    owner=scalar
                    values=tuple(value[0] for value in values)
                graph=owner._specialized_autodiff_module(values,{})
                program=candidate.lower_typed_scaled(graph)
                from tessera.compiler import rocm_native,rocm_fp8_blockscale
                for phase in ("empty_image_caches","reused_image_caches"):
                    samples={"control":[],"candidate":[]}
                    counts={"control":[],"candidate":[]}
                    for sample in range(3):
                        order=("control","candidate") if sample%2==0 else ("candidate","control")
                        for arm in order:
                            if phase=="empty_image_caches":
                                rocm_native._cache.clear()
                                rocm_fp8_blockscale._blockscale_target_cache.clear()
                            adapter=control if arm=="control" else candidate
                            original=subprocess.run
                            count=[0]
                            def observed(*args,**kwargs):
                                count[0]+=1
                                return original(*args,**kwargs)
                            subprocess.run=observed
                            try:
                                start=time.perf_counter()
                                package=adapter.package_typed_scaled(graph,program,
                                    pipeline_name="tessera-lower-to-rocm")
                                samples[arm].append((time.perf_counter()-start)*1e3)
                            finally:
                                subprocess.run=original
                            counts[arm].append(count[0])
                            native=NativeScaledProgram.from_manifest(
                                package.descriptor.provenance["native_scaled_primal_program"])
                            native.validate()
                            if arm=="candidate":
                                assert package.image.payload==native.images[0]
                    medians={arm:statistics.median(v) for arm,v in samples.items()}
                    rows.append({"policy":policy,"format":fmt,"transpose_b":nk,
                        "cache_phase":phase,"package_samples_ms":samples,
                        "package_medians_ms":medians,"subprocess_counts":counts,
                        "control_over_candidate":medians["control"]/medians["candidate"]})
    tool=candidate.find_tessera_opt()
    return {"architecture":runtime._rocm_live_arch(),
        "measurement":"package_wall_time_excluding_graph_to_tile_with_explicit_image_cache_phases",
        "control_sha256":hashlib.sha256(Path(control_path).read_bytes()).hexdigest(),
        "candidate_sha256":hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        "compiler_sha256":hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
        "rows":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--control",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.control),indent=2)+"\n")
