"""Correctness-gated native scale-VJP member/program baseline on gfx1201."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
import numpy as np
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram, package_native_scaled_vjp
from tests.unit.test_native_scaled_transpose_export import source
from tests.unit.test_native_nested_typed_vmap import nested_case
from tests.support.scaled_product_transpose_oracle import scale_adjoint

POLICIES = ("shared_rhs_rows", "independent_rhs", "shared_lhs")
REQUESTS = ((2,), (3,), (2, 3), (3, 2))


def key(policy, nk, roles):
    return policy+("_nk_" if nk else "_kn_")+"".join(map(str, roles))


def values_for_shape(policy, nk, shape):
    # Diagnostic input construction only; native packages own all arithmetic.
    import ml_dtypes
    b0,b1,m,n,k = shape
    prefix = (b0,b1)
    ap = prefix if policy != "shared_lhs" else ()
    bp = prefix if policy != "shared_rhs_rows" else ()
    rng = np.random.default_rng(1007)
    a = rng.choice([-.5,0,.25,1],ap+(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b = rng.choice([-1,0,.5,2],bp+(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b = np.ascontiguousarray(logical_b.swapaxes(-1,-2)) if nk else logical_b
    g,c = (k+127)//128,(n+127)//128
    sa = rng.uniform(.2,1,ap+(m,g)).astype(np.float32)
    sb = rng.uniform(.2,1,bp+(g,c)).astype(np.float32)
    return (a,b,sa,sb)

def emit_packages(directory, *, wave=False, shape=(2,3,7,19,256)):
    directory.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        for nk in (False, True):
            for roles in REQUESTS:
                graph = source(policy, roles, nk, shape=shape)
                package = package_native_scaled_vjp(graph, schedule="wave_per_scale_element" if wave else "serial_per_scale_element")
                (directory/(key(policy, nk, roles)+".json")).write_text(json.dumps(package.to_manifest())+"\n")


def measure(directory):
    from tessera import runtime
    arch = runtime._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError("native scale-VJP baseline requires actual gfx1201")
    library = runtime._load_rocm_native_movement_runtime()._name
    rows = []
    for policy in POLICIES:
        for nk in (False, True):
            _, _, _, values, _ = nested_case(policy, "fp32", nk)
            dy = np.random.default_rng(1007).uniform(-.5, .5, (2, 3, 7, 19)).astype(np.float32)
            gradients = scale_adjoint(*values, dy, scale_k=128, scale_n=128,
                                      batching=policy, transpose_b=nk)
            for roles in REQUESTS:
                package = NativeScaledProgram.from_manifest(json.loads(
                    (directory/(key(policy, nk, roles)+".json")).read_text()))
                # Replay must not invoke the compiler or production reference.
                original_run = subprocess.run
                def forbidden(*args, **kwargs):
                    raise AssertionError("serialized scale-VJP replay invoked compiler")
                subprocess.run = forbidden
                try:
                    with PreparedScaledProgram(package, [*values, dy], runtime_library=library) as owner:
                        generation, _ = owner.invoke()
                        actual = owner.read(generation)
                        errors = []
                        for got, role in zip(actual, roles, strict=True):
                            expected = gradients[role-2]
                            np.testing.assert_allclose(got, expected, rtol=4e-5, atol=2e-4)
                            errors.append(float(np.max(np.abs(got-expected))))
                        owner.update([*values, dy*-.5])
                        new_generation, _ = owner.invoke()
                        assert new_generation != generation
                        changed = owner.read(new_generation)
                        for got, role in zip(changed, roles, strict=True):
                            np.testing.assert_allclose(got, gradients[role-2]*-.5,
                                                       rtol=4e-5, atol=2e-4)
                        try:
                            owner.read(generation)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError("stale native gradient generation remained readable")
                        owner.update([*values, dy])
                        events, host = [], []
                        for _ in range(3):
                            last, elapsed = owner.invoke(repeats=10, timed=True)
                            events.append(elapsed)
                            start = time.perf_counter()
                            for _ in range(5):
                                owner.update([*values, dy])
                                last, _ = owner.invoke()
                                owner.read(last)
                            host.append((time.perf_counter()-start)*1000/5)
                        for got, role in zip(owner.read(last), roles, strict=True):
                            np.testing.assert_allclose(got, gradients[role-2], rtol=4e-5, atol=2e-4)
                finally:
                    subprocess.run = original_run
                rows.append({"policy":policy, "rhs_nk":nk, "gradient_arguments":list(roles),
                    "shape_b0b1mnk":[2,3,7,19,256], "native_members":len(package.images),
                    "scale_adjoint_schedule":json.loads(package.members_json[0]).get("scale_adjoint_schedule","serial_per_scale_element"),
                    "correctness":"passed_before_and_after_timing",
                    "max_abs_error":errors,
                    "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
                    "native_event_samples_ms":events, "native_event_median_ms":median(events),
                    "prepared_update_invoke_read_samples_ms":host,
                    "prepared_update_invoke_read_median_ms":median(host)})
    return {"schema":1, "live_architecture":arch, "runtime_library":library,
        "route":"typed Graph native paired AD -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM -> HSACO -> checked native program ABI",
        "timing_boundary":"one-member or two-member native HIP launch-window events; prepared host update/invoke/read measured separately",
        "limitations":["initial serial-per-scale-element baseline", "public Python JIT reverse API not yet connected",
                       "no performance promotion", "no sibling architecture proof"],
        "rows":rows}


def measure_public():
    import tessera as ts
    from tessera import runtime
    from tessera.autodiff import vmap
    from tests.unit.test_native_typed_scaled_vmap import case
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("public scale-VJP baseline requires actual gfx1201")
    rows = []
    for policy in POLICIES:
        for nk in (False, True):
            for depth in (1, 2):
                scalar, mapped, values, expected = case(policy, "fp32", nk)
                if depth == 2:
                    _, _, _, values, expected = nested_case(policy, "fp32", nk)
                dy = np.random.default_rng(1007).uniform(-.5, .5, expected.shape).astype(np.float32)
                oracle = scale_adjoint(*values, dy, scale_k=128, scale_n=128,
                                       batching=policy, transpose_b=nk)
                for roles in REQUESTS:
                    names = tuple("sa" if role == 2 else "sb" for role in roles)
                    owner = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=names)(scalar._fn)
                    for _ in range(depth):
                        owner = vmap(owner, in_axes=mapped._frontend_batch_axes)
                    got = owner.native_backward(*values, out_cotangents=dy)
                    for output, role in zip(got, roles, strict=True):
                        np.testing.assert_allclose(output, oracle[role-2], rtol=4e-5, atol=2e-4)
                    original_run = subprocess.run
                    def forbidden(*args, **kwargs):
                        raise AssertionError("warm public VJP invoked compiler")
                    subprocess.run = forbidden
                    try:
                        samples = []
                        for _ in range(3):
                            start = time.perf_counter()
                            for _ in range(5):
                                got = owner.native_backward(*values, out_cotangents=dy)
                            samples.append((time.perf_counter()-start)*1000/5)
                        for output, role in zip(got, roles, strict=True):
                            np.testing.assert_allclose(output, oracle[role-2], rtol=4e-5, atol=2e-4)
                    finally:
                        subprocess.run = original_run
                    receipt = owner.last_backward_execution
                    assert receipt["execution_certificate"]["evidence_scope"] == "exact_device"
                    rows.append({"policy":policy, "rhs_nk":nk, "map_depth":depth,
                        "gradient_arguments":list(roles), "correctness":"passed_before_and_after_timing",
                        "public_call_samples_ms":samples, "public_call_median_ms":median(samples),
                        "artifact_hash":receipt["artifact_hash"],
                        "execution_certificate":receipt["execution_certificate"]})
    return {"schema":1, "live_architecture":"gfx1201",
        "timing_boundary":"compiler-free warm public native_backward wall time; includes frontend, package/ABI, upload, native launches and readback",
        "limitations":["static leading maps and FP32 scale gradients", "initial serial reduction", "no isolated-kernel or speedup claim"],
        "rows":rows}


def measure_paired(serial_directory, wave_directory, *, shape=(2,3,7,19,256), repeats=10):
    from contextlib import ExitStack
    from tessera import runtime
    from tessera.compiler.native_scaled_program import PreparedScaledProgram
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("paired scale-VJP baseline requires actual gfx1201")
    library = runtime._load_rocm_native_movement_runtime()._name
    rows = []
    for policy in POLICIES:
        for nk in (False, True):
            values = values_for_shape(policy, nk, shape)
            dy = np.random.default_rng(1007).uniform(-.5,.5,tuple(shape[:2])+tuple(shape[2:4])).astype(np.float32)
            oracle = scale_adjoint(*values, dy, scale_k=128, scale_n=128,
                                   batching=policy, transpose_b=nk)
            for roles in REQUESTS:
                with ExitStack() as stack:
                    owners, images = {}, {}
                    for name, directory in (("serial",serial_directory),("wave",wave_directory)):
                        package = NativeScaledProgram.from_manifest(json.loads(
                            (directory/(key(policy,nk,roles)+".json")).read_text()))
                        images[name] = [hashlib.sha256(image).hexdigest() for image in package.images]
                        owners[name] = stack.enter_context(PreparedScaledProgram(
                            package,[*values,dy],runtime_library=library))
                    original_run = subprocess.run
                    def forbidden(*args,**kwargs):
                        raise AssertionError("paired native replay invoked compiler")
                    subprocess.run = forbidden
                    try:
                        errors = {}
                        for name, owner in owners.items():
                            generation,_ = owner.invoke()
                            outputs = owner.read(generation)
                            errors[name] = []
                            for got, role in zip(outputs,roles,strict=True):
                                want = oracle[role-2]
                                np.testing.assert_allclose(got,want,rtol=4e-5,atol=2e-4)
                                errors[name].append(float(np.max(np.abs(got-want))))
                        samples = {"serial":[], "wave":[]}
                        for window in range(6):
                            order = ("serial","wave") if window%2==0 else ("wave","serial")
                            for name in order:
                                _,elapsed = owners[name].invoke(repeats=repeats,timed=True)
                                samples[name].append(elapsed)
                        for name,owner in owners.items():
                            owner.update([*values,dy*-.5])
                            generation,_ = owner.invoke()
                            for got,role in zip(owner.read(generation),roles,strict=True):
                                np.testing.assert_allclose(got,oracle[role-2]*-.5,rtol=4e-5,atol=2e-4)
                    finally:
                        subprocess.run = original_run
                rows.append({"shape_b0b1mnk":list(shape),"policy":policy,"rhs_nk":nk,"gradient_arguments":list(roles),
                    "correctness":"passed_before_and_after_timing","max_abs_error":errors,
                    "image_sha256":images,"event_samples_ms":samples,
                    "serial_median_ms":median(samples["serial"]),"wave_median_ms":median(samples["wave"]),
                    "serial_over_wave":median(samples["serial"])/median(samples["wave"])})
    return {"schema":1,"live_architecture":"gfx1201","rows":rows,
        "timing_boundary":"alternating native one/two-member HIP event launch windows; includes native submission work, excludes Python enqueue loops and host transfer/readback",
        "limitations":["static FP32 scale-adjoint profile","different native allocations per arm","no isolated ISA clock or production selector promotion"]}



def measure_public_paired(*, shape=(2,3,7,19,256)):
    import os
    import tessera as ts
    from tessera import runtime
    from tessera.autodiff import vmap
    from tests.unit.test_native_typed_scaled_vmap import case
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("public paired scale-VJP requires actual gfx1201")
    option = "TESSERA_ROCM_SCALE_VJP_SCHEDULE"
    prior = os.environ.get(option)
    schedules = ("serial_per_scale_element", "wave_per_scale_element")
    rows = []
    try:
        for policy in POLICIES:
            for nk in (False, True):
                for depth in (1, 2):
                    scalar, mapped, values, expected = case(policy, "fp32", nk,
                        (shape[0], *shape[2:]))
                    if depth == 2:
                        _, _, _, values, expected = nested_case(policy, "fp32", nk, shape=shape)
                    dy = np.random.default_rng(1007).uniform(-.5,.5,expected.shape).astype(np.float32)
                    oracle = scale_adjoint(*values,dy,scale_k=128,scale_n=128,
                                           batching=policy,transpose_b=nk)
                    for roles in REQUESTS:
                        owners, hashes, errors = {}, {}, {}
                        names = tuple("sa" if role==2 else "sb" for role in roles)
                        def check(got, factor=1):
                            for output, role in zip(got,roles,strict=True):
                                np.testing.assert_allclose(output,oracle[role-2]*factor,
                                                           rtol=4e-5,atol=2e-4)
                        for schedule in schedules:
                            os.environ[option] = schedule
                            owner = ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=names)(scalar._fn)
                            for _ in range(depth):
                                owner = vmap(owner,in_axes=mapped._frontend_batch_axes)
                            got = owner.native_backward(*values,out_cotangents=dy)
                            check(got)
                            receipt = owner.last_backward_execution
                            assert receipt["scale_adjoint_schedule"] == schedule
                            assert receipt["execution_certificate"]["evidence_scope"] == "exact_device"
                            owners[schedule] = owner
                            hashes[schedule] = receipt["artifact_hash"]
                            errors[schedule] = [float(np.max(np.abs(output-oracle[role-2])))
                                                for output,role in zip(got,roles,strict=True)]
                        assert hashes[schedules[0]] != hashes[schedules[1]]
                        original_run = subprocess.run
                        def forbidden(*args,**kwargs):
                            raise AssertionError("warm paired public VJP invoked compiler")
                        subprocess.run = forbidden
                        samples = {schedule:[] for schedule in schedules}
                        try:
                            for window in range(6):
                                order = schedules if window%2==0 else schedules[::-1]
                                for schedule in order:
                                    os.environ[option] = schedule
                                    start = time.perf_counter()
                                    for _ in range(3):
                                        got = owners[schedule].native_backward(*values,out_cotangents=dy)
                                    samples[schedule].append((time.perf_counter()-start)*1000/3)
                                    check(got)
                            for schedule in schedules:
                                os.environ[option] = schedule
                                check(owners[schedule].native_backward(*values,out_cotangents=dy*-.5),-.5)
                        finally:
                            subprocess.run = original_run
                        medians = {schedule:median(samples[schedule]) for schedule in schedules}
                        rows.append({"policy":policy,"rhs_nk":nk,"map_depth":depth,
                            "output_shape":list(expected.shape),"shape_b0b1mnk":list(shape),
                            "gradient_arguments":list(roles),"max_abs_error":errors,
                            "correctness":"passed_before_and_after_timing",
                            "artifact_hashes":hashes,"public_samples_ms":samples,
                            "public_medians_ms":medians,
                            "serial_over_wave":medians[schedules[0]]/medians[schedules[1]]})
    finally:
        if prior is None:
            os.environ.pop(option,None)
        else:
            os.environ[option] = prior
    return {"schema":1,"live_architecture":"gfx1201","rows":rows,
        "timing_boundary":"alternating compiler-free public native_backward wall time including frontend, ABI, upload, native launch and readback",
        "limitations":["static FP32 scale adjoints and one/two leading maps",
                       "no isolated-kernel speedup or selector promotion"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=5, default=[2,3,7,19,256])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compare-wave-packages", type=Path)
    parser.add_argument("--wave", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--public-compare-wave", action="store_true")
    parser.add_argument("--emit-packages", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(x <= 0 for x in args.shape) or not 0 < args.repeats <= 1048576:
        parser.error("shape extents and bounded repetition count must be positive")
    if args.compare_wave_packages:
        if not args.packages or not args.output:
            parser.error("--packages and --output are required for paired comparison")
        args.output.write_text(json.dumps(measure_paired(args.packages,args.compare_wave_packages,shape=tuple(args.shape),repeats=args.repeats),indent=2)+"\n")
    elif args.public_compare_wave:
        if not args.output:
            parser.error("--output is required for public paired measurement")
        args.output.write_text(json.dumps(measure_public_paired(shape=tuple(args.shape)),indent=2)+"\n")
    elif args.public:
        if not args.output:
            parser.error("--output is required for public measurement")
        args.output.write_text(json.dumps(measure_public(), indent=2)+"\n")
    elif args.emit_packages:
        emit_packages(args.emit_packages, wave=args.wave,shape=tuple(args.shape))
    else:
        if not args.packages or not args.output:
            parser.error("--packages and --output are required for owning-device measurement")
        args.output.write_text(json.dumps(measure(args.packages), indent=2)+"\n")


if __name__ == "__main__":
    main()
