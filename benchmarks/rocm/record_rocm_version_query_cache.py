"""A/B actual package work with version metadata cold versus reused."""
import argparse,hashlib,json,statistics,subprocess,time
from pathlib import Path
from tessera import runtime
from tessera.compiler import rocm_native as native
from tessera.compiler.rocm_typed_scaled_native import lower_typed_scaled,package_typed_scaled
from tests.unit.test_native_typed_scaled_vmap import case

def measure():
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[]
    for policy in (None,"shared_rhs_rows","independent_rhs","shared_lhs"):
        for fmt in ("fp32","e8m0"):
            for nk in (False,True):
                scalar,owner,values,_=case(policy or "independent_rhs",fmt,nk)
                if policy is None:
                    owner=scalar;values=tuple(v[0] for v in values)
                graph=owner._specialized_autodiff_module(values,{})
                program=lower_typed_scaled(graph)
                # Prime stable compiler/driver metadata outside both timings.
                package_typed_scaled(graph,program,pipeline_name="tessera-lower-to-rocm")
                samples={arm:[] for arm in ("uncached_versions","cached_versions")}
                counts={arm:[] for arm in samples}
                identities={}
                for index in range(5):
                    order=tuple(samples) if index%2==0 else tuple(reversed(samples))
                    for arm in order:
                        preserved=native._VERSION_FINGERPRINTS
                        if arm=="uncached_versions":native._VERSION_FINGERPRINTS={}
                        original=subprocess.run;calls=[0]
                        def observed(*args,**kwargs):
                            calls[0]+=1;return original(*args,**kwargs)
                        subprocess.run=observed
                        try:
                            start=time.perf_counter()
                            package=package_typed_scaled(graph,program,pipeline_name="tessera-lower-to-rocm")
                            samples[arm].append((time.perf_counter()-start)*1e3)
                        finally:
                            subprocess.run=original;native._VERSION_FINGERPRINTS=preserved
                        counts[arm].append(calls[0])
                        identities[arm]=(package.image.compiler_fingerprint,package.image.toolchain_fingerprint)
                assert identities["uncached_versions"]==identities["cached_versions"]
                medians={arm:statistics.median(v) for arm,v in samples.items()}
                rows.append({"policy":policy,"format":fmt,"transpose_b":nk,
                    "samples_ms":samples,"medians_ms":medians,"subprocess_counts":counts,
                    "uncached_over_cached":medians["uncached_versions"]/medians["cached_versions"],
                    "fingerprints_match":True})
                print("passed",len(rows),"/16",flush=True)
    return {"architecture":runtime._rocm_live_arch(),
        "measurement":"real native Graph-derived package wall time; version metadata only is varied",
        "source_sha256":hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256":hashlib.sha256(Path(native._tessera_opt()).read_bytes()).hexdigest(),
        "limitations":["package timings, not kernel timings","static scalar/coupled profiles","no selector promotion"],
        "rows":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(measure(),indent=2)+"\n")
