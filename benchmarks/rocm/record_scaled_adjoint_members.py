"""Own gfx1201 FP8 scale-adjoint member cost attribution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import numpy as np
import tessera as ts
from tessera import runtime
from tessera.autodiff import vmap
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.unit.test_native_typed_scaled_vmap import case
from tests.device.rocm.test_public_mapped_scale_vjp import scale_oracle


def record(roles):
    scalar, leading, raw, primal = case("independent_rhs", "fp32", False, (2,7,19,256))
    owner=vmap(ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(scalar._fn),
               in_axes=leading._frontend_batch_axes)
    seed=np.random.default_rng(941).uniform(-.5,.5,primal.shape).astype(np.float32)
    reference=scale_oracle(raw,leading._frontend_batch_axes,seed)
    expected=tuple(reference[{"sa":0,"sb":1}[role]] for role in roles)
    def compare(outputs):
        for actual, wanted in zip(outputs,expected,strict=True):
            np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=3e-5)
    compare(owner.native_backward(*raw,out_cotangents=seed))
    receipt=owner.last_backward_execution
    if receipt["execution_kind"]!="native_gpu" or receipt["evidence_target"]!="rocm_gfx1201":
        raise RuntimeError("missing exact-device native execution receipt")
    package=owner._native_backward_artifact
    captured=[]; interleaved=[]
    with PreparedScaledProgram(package,(*raw,seed),
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        generation,_=prepared.invoke();compare(prepared.read(generation))
        for _ in range(5):
            generation,members=prepared.profile_members(repeats=512)
            compare(prepared.read(generation));captured.append(list(members))
            generation,elapsed=prepared.invoke(repeats=128,timed=True)
            compare(prepared.read(generation));interleaved.append(elapsed)
    program=json.loads(package.program_json)
    return {"roles":roles,"shape_bmnk":[2,7,19,256],
        "steps":program["steps"],"buffers":program["buffers"],
        "captured_member_samples_ms":captured,
        "captured_member_median_ms":[median(row[i] for row in captured)
            for i in range(len(captured[0]))],
        "interleaved_samples_ms":interleaved,"interleaved_median_ms":median(interleaved),
        "captured_repetitions":512,"ordinary_repetitions":128,
        "program_sha256":hashlib.sha256(package.program_json.encode()).hexdigest(),
        "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
        "correctness":"independent_before_and_after_each_captured_and_ordinary_window"}


def record_schedule_pair(roles, shape, include_auto=False):
    from contextlib import ExitStack
    from tessera.compiler.native_scaled_program import package_native_scaled_vjp
    scalar,leading,raw,primal=case("independent_rhs","fp32",False,shape)
    owner=vmap(ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(scalar._fn),
               in_axes=leading._frontend_batch_axes)
    seed=np.random.default_rng(941).uniform(-.5,.5,primal.shape).astype(np.float32)
    reference=scale_oracle(raw,leading._frontend_batch_axes,seed)
    expected=tuple(reference[{"sa":0,"sb":1}[role]] for role in roles)
    def compare(outputs):
        for actual,wanted in zip(outputs,expected,strict=True):
            np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=3e-5)
    compare(owner.native_backward(*raw,out_cotangents=seed))
    serial=owner._native_backward_artifact
    wave=package_native_scaled_vjp(serial.graph_ir,schedule="wave_per_scale_element")
    packages={"serial":serial,"wave":wave}
    if include_auto:packages["auto"]=package_native_scaled_vjp(serial.graph_ir,schedule="auto")
    assert all(package.program_json==serial.program_json for package in packages.values())
    samples={name:[] for name in packages};members={name:[] for name in packages}
    with ExitStack() as stack:
        owners={name:stack.enter_context(PreparedScaledProgram(package,(*raw,seed),
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]))
            for name,package in packages.items()}
        for round_ in range(7):
            names=tuple(owners)
            pivot=round_%len(names)
            order=names[pivot:]+names[:pivot]
            if round_%2:order=tuple(reversed(order))
            for name in order:
                prepared=owners[name]
                generation,_=prepared.invoke();compare(prepared.read(generation))
                generation,elapsed=prepared.invoke(repeats=128,timed=True)
                compare(prepared.read(generation));samples[name].append(elapsed)
                generation,elapsed=prepared.profile_members(repeats=512)
                compare(prepared.read(generation));members[name].append(list(elapsed))
    medians={key:median(value) for key,value in samples.items()}
    return {"roles":roles,"shape_bmnk":shape,"ordinary_samples_ms":samples,
        "ordinary_median_ms":medians,"wave_over_serial":medians["wave"]/medians["serial"],
        "captured_member_samples_ms":members,
        "captured_member_median_ms":{key:[median(row[i] for row in value)
            for i in range(len(value[0]))] for key,value in members.items()},
        "correctness":"independent_before_and_after_each_alternating_arm",
        "image_sha256":{name:[hashlib.sha256(image).hexdigest() for image in package.images]
            for name,package in packages.items()},
        "member_algorithms":{name:[json.loads(raw)["scale_adjoint_schedule"] for raw in package.members_json] for name,package in packages.items()},
        "program_sha256":{name:hashlib.sha256(package.program_json.encode()).hexdigest()
            for name,package in packages.items()}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--paired",action="store_true")
    parser.add_argument("--auto",action="store_true")
    args=parser.parse_args()
    arch=runtime._rocm_live_arch()
    if arch!="gfx1201":raise RuntimeError("owning gfx1201 required")
    paths=[Path(os.environ[name]).resolve() for name in (
        "TESSERA_OPT","TESSERA_ROCM_OPT","TESSERA_ROCM_NATIVE_MOVEMENT_LIB")]
    paths.extend(Path(name).resolve() for name in (
        "src/compiler/ir/LinearTransposeInterface.cpp",
        "src/compiler/programming_model/lib/NativeScaleTranspose.h",
        "src/compiler/programming_model/lib/PMPasses.cpp",
        "python/tessera/compiler/native_vjp_plugins.py",
        "python/tessera/compiler/pass_metadata.py",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "python/tessera/compiler/native_scaled_program.py",
        "tests/unit/test_native_typed_scaled_vmap.py",
        "tests/device/rocm/test_public_mapped_scale_vjp.py"))
    paths.append(Path(__file__).resolve())
    packet={"architecture":arch,
        "device_inventory":subprocess.run(["rocminfo"],check=True,capture_output=True,
            text=True,timeout=30).stdout,
        "identity_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases":([record_schedule_pair(roles,shape,args.auto) for shape in ((2,3,5,37),(2,7,19,256),(2,17,19,256)) for roles in (("sa",),("sb",),("sa","sb"),("sb","sa"))] if args.paired else [record(roles) for roles in (("sa",),("sb",),("sa","sb"),("sb","sa"))]),
        "scope":"captured pure SSA member graph windows include device graph dispatch; ordinary whole-program events include host dispatch gaps; neither includes host transfers or compilation"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":main()
