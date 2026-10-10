"""Scalar scale reverse baseline with separate public and native timing."""
import argparse
import ctypes as c
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.unit.test_public_independent_scale_vmap import case
from benchmarks.rocm.record_independent_batch_reverse_device import oracle

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(actual, wanted):
    error = 0.
    for got, expected in zip(actual, wanted, strict=True):
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-5)
        error = max(error, float(np.max(np.abs(got-expected))))
    return error

def measure():
    arch = rt._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError("public independent-scale proof requires actual gfx1201")
    library = Path(rt._load_rocm_native_movement_runtime()._name)
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    if hip.hipInit(0):
        raise RuntimeError("HIP initialization failed")
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte*16)()
    version, driver = c.c_int(), c.c_int()
    if (hip.hipDeviceGetName(name,256,0) or hip.hipDeviceGetUuid(c.byref(uuid),0)
        or hip.hipRuntimeGetVersion(c.byref(version))
        or hip.hipDriverGetVersion(c.byref(driver))):
        raise RuntimeError("HIP device identity query failed")
    rows = []
    for prefix, mask, ta, tb in itertools.product(
        ((),), (1,), (False,True), (False,True)):
        scalar, owner, values, axes = case(mask,ta,tb,prefix)
        _, _, changed, _ = case(mask,ta,tb,prefix,seed=5007)
        cot = np.random.default_rng(1007).uniform(-1,1,(*prefix,3,5)).astype(np.float32)
        row = {"output_prefix":prefix,"transposeA":ta,"transposeB":tb}
        wanted = oracle((*values,cot),row)
        maximum = check(owner.native_backward(*values,out_cotangents=cot),wanted)
        package = owner.native_backward_runtime_artifact()
        receipt = owner.last_backward_execution
        assert receipt["compiler_path"] == "rocm_scaled_vjp_program_compiled"
        assert receipt["execution_certificate"]["evidence_scope"] == "exact_device"
        assert receipt["physical_attestation"]["device_arch"] == arch
        manifest = json.dumps(package.to_manifest(),sort_keys=True,separators=(",",":"))
        original_run = subprocess.run
        def forbidden(*args,**kwargs):
            raise AssertionError("warm reverse measurement invoked compiler subprocess")
        subprocess.run = forbidden
        try:
            cot = -.75*cot
            wanted = oracle((*changed,cot),row)
            maximum = max(maximum,check(owner.native_backward(*changed,out_cotangents=cot),wanted))
            public = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(5):
                    actual = owner.native_backward(*changed,out_cotangents=cot)
                public.append((time.perf_counter()-start)*1e3/5)
                maximum = max(maximum,check(actual,wanted))
            events, prepared_host = [], []
            with PreparedScaledProgram(package,[*changed,cot],runtime_library=str(library)) as prepared:
                old, _ = prepared.invoke()
                maximum = max(maximum,check(prepared.read(old),wanted))
                for _ in range(5):
                    generation, elapsed = prepared.invoke(repeats=20,timed=True)
                    events.append(elapsed)
                    maximum = max(maximum,check(prepared.read(generation),wanted))
                    start = time.perf_counter()
                    for _ in range(5):
                        prepared.update([*changed,cot])
                        generation, _ = prepared.invoke()
                        actual = prepared.read(generation)
                    prepared_host.append((time.perf_counter()-start)*1e3/5)
                    maximum = max(maximum,check(actual,wanted))
                try:
                    prepared.read(old)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("stale native output generation accepted")
        finally:
            subprocess.run = original_run
        assert scalar._frontend_batch_axes is None
        rows.append({"mask":mask,"axes":axes,**row,
            "correctness":"passed_before_and_after_each_timing_domain",
            "changed_input_replay":"passed","compiler_free_warm_replay":True,
            "stale_generation":"rejected","max_abs_error":maximum,
            "public_warm_call_samples_ms":public,"public_warm_call_median_ms":median(public),
            "native_launch_window_samples_ms":events,"native_launch_window_median_ms":median(events),
            "prepared_update_invoke_read_samples_ms":prepared_host,
            "prepared_update_invoke_read_median_ms":median(prepared_host),
            "package_sha256":hashlib.sha256(manifest.encode()).hexdigest(),
            "images_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
            "execution_certificate":receipt["execution_certificate"]})
        print("passed",len(rows),"/ 4",flush=True)
    from tessera.compiler import native_vmap,rocm_typed_scaled_native,native_scaled_program
    return {"schema":1,"architecture":arch,"device":name.value.decode(),
        "device_uuid_raw_hex":bytes(uuid).hex(),
        "hip_runtime_version":version.value,"hip_driver_version":driver.value,
        "compiler_sha256":digest(__import__("os").environ["TESSERA_OPT"]),
        "runtime_sha256":digest(library),"recorder_sha256":digest(__file__),
        "adapter_source_sha256":{m.__name__:digest(m.__file__) for m in
            (rt,native_vmap,rocm_typed_scaled_native,native_scaled_program)},
        "route":"public scalar native_backward -> typed Graph paired AD -> Schedule -> Tile structured reduction -> ROCm Target -> LLVM -> HSACO -> checked HIP owner",
        "timing_boundaries":{
            "public":"compiler-free warm native_backward wall time including frontend, ABI, upload, launch, readback and owner disposal",
            "native":"two-member HIP event launch window averaged by native runtime over 20 repeats; no isolated-kernel claim",
            "prepared":"native owner update/invoke/read host wall time; preparation excluded"},
        "limitations":["tiny static M3 N5 K7 baseline","FP32 scale gradients only",
            "K4 scale groups are reverse-only in this packet; primal WMMA requires its own admitted group width","no speedup or selector promotion",
            "no sibling architecture proof"],"rows":rows}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = measure()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")

if __name__ == "__main__":
    main()
