"""Owning continuous mapped reverse: independent numerics and separate costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch
import numpy as np
from tessera import runtime
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tessera.compiler.native_vmap import mixed_batch_policies, normalize_mixed_batch_inputs
from tests.unit.test_public_floating_scaled_maps import mapped_case, mixed_case
from tests.device.rocm.test_floating_scaled_adjoint import batch_oracle


def record(label, mixed=False, mask=15, ta=False, tb=False, prefix=(2,3), out_axes=0, repetitions=2048):
    start=time.perf_counter()
    if mixed:
        _,owner,raw,seed,expected=mixed_case(ta,tb,True)
        native_expected=(expected[0],np.expand_dims(np.moveaxis(expected[1],1,0),1),
                         np.expand_dims(expected[2],0),
                         np.expand_dims(np.moveaxis(expected[3],-1,0),1))
    else:
        _,owner,raw,seed=mapped_case(mask,ta,tb,prefix,out_axes)
        expected=batch_oracle((*raw,seed),ta,tb)
        native_expected=expected
    setup=(time.perf_counter()-start)*1e3
    mapped_seed=np.ascontiguousarray(seed.transpose(owner._frontend_output_permutation))
    def compare(outputs, references):
        for output,reference in zip(outputs,references,strict=True):
            np.testing.assert_allclose(output,reference,rtol=4e-5,atol=3e-6)
    start=time.perf_counter()
    actual=owner.native_backward(*raw,out_cotangents=mapped_seed)
    first=(time.perf_counter()-start)*1e3
    compare(actual,expected)
    receipt=owner.last_backward_execution
    if (receipt["execution_kind"]!="native_gpu" or receipt["frontend_authority"]!="tracer"
            or receipt["evidence_target"]!="rocm_gfx1201"):
        raise RuntimeError("public mapped reverse lacks owning compiler receipt")
    package=owner._native_backward_artifact
    def forbidden(*a,**kw): raise AssertionError("warm mapped reverse invoked compiler")
    public=[]
    with patch("subprocess.run",side_effect=forbidden):
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(20):
                actual=owner.native_backward(*raw,out_cotangents=mapped_seed)
            public.append((time.perf_counter()-start)*1e3/20)
            compare(actual,expected)
    normalized=(normalize_mixed_batch_inputs(raw,owner._frontend_batch_policies)
                if mixed_batch_policies(owner) else raw)
    captured=[];native=[]
    with PreparedScaledProgram(package,[*normalized,mapped_seed],
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        generation,_=prepared.invoke()
        compare(prepared.read(generation),native_expected)
        for _ in range(5):
            generation,members=prepared.profile_members(repeats=repetitions)
            captured.append(list(members))
            compare(prepared.read(generation),native_expected)
            generation,elapsed=prepared.invoke(repeats=128,timed=True)
            native.append(elapsed)
            compare(prepared.read(generation),native_expected)
    program=json.loads(package.program_json)
    return {"label":label,"raw_shapes":[list(value.shape) for value in raw],
        "normalized_shapes":[list(value.shape) for value in normalized],
        "map_policies":owner._frontend_batch_policies,
        "output_permutation":owner._frontend_output_permutation,
        "mapped_seed_shape":list(mapped_seed.shape),
        "transpose_a":ta,"transpose_b":tb,"gradient_roles":program["gradient_roles"],
        "members":[row.get("gradient_role",row["operation"]) for row in program["steps"]],
        "correctness":"independent_float64_before_and_after_each_timing_domain",
        "max_abs_error":max(float(np.max(np.abs(a-b))) for a,b in zip(actual,expected,strict=True)),
        "setup_with_fixture_and_oracle_ms":setup,"first_public_reverse_ms":first,
        "warm_public_reverse_samples_ms":public,"warm_public_reverse_median_ms":median(public),
        "captured_repetitions":repetitions,"captured_member_samples_ms":captured,
        "captured_member_median_ms":[median(row[i] for row in captured) for i in range(len(captured[0]))],
        "captured_min_window_ms":min(min(row)*repetitions for row in captured),
        "interleaved_native_program_samples_ms":native,
        "interleaved_native_program_median_ms":median(native),
        "source_graph_sha256":hashlib.sha256(owner._traced_autodiff_module(raw,{}).to_mlir(target="rocm_gfx1201").encode()).hexdigest(),
        "native_package_sha256":hashlib.sha256(package.program_json.encode()+b"".join(package.images)).hexdigest(),
        "physical_attestation":receipt["physical_attestation"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=2048)
    args=parser.parse_args()
    if args.repetitions<=0: parser.error("repetitions must be positive")
    arch=runtime._rocm_live_arch()
    if arch!="gfx1201": raise RuntimeError("owning gfx1201 required")
    source_paths=[
        "python/tessera/autodiff/transforms.py","python/tessera/compiler/jit.py",
        "python/tessera/compiler/native_vmap.py","python/tessera/compiler/rocm_typed_scaled_native.py",
        "python/tessera/compiler/native_vjp_plugins.py","python/tessera/compiler/native_scaled_program.py",
        "python/tessera/compiler/graph_ir.py","python/tessera/compiler/capabilities.py",
        "python/tessera/compiler/reference_typed_scaled_matmul.py",
        "src/compiler/ir/LinearTransposeInterface.cpp","src/compiler/ir/TesseraOps.cpp",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
        "tests/unit/test_public_floating_scaled_maps.py",
        "tests/unit/test_public_floating_scaled_reverse.py",
        "tests/device/rocm/test_public_floating_scaled_maps.py",
        "tests/device/rocm/test_floating_scaled_adjoint.py",
    ]
    paths=[Path(os.environ[name]).resolve() for name in (
        "TESSERA_OPT","TESSERA_ROCM_OPT","TESSERA_ROCM_NATIVE_MOVEMENT_LIB")]
    paths.extend(Path(path).resolve() for path in source_paths)
    paths.append(Path(__file__).resolve())
    packet={"architecture":arch,
        "device_inventory":subprocess.run(["rocminfo"],check=True,capture_output=True,text=True,timeout=30).stdout,
        "identity_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases":[record("shared_rhs_leading",mask=5,prefix=(2,),repetitions=args.repetitions),
                 record("independent_matrix_scale_nonleading",mask=6,tb=True,out_axes=-1,repetitions=args.repetitions),
                 record("all_mapped_transposed_nonleading",mask=15,ta=True,tb=True,out_axes=-1,repetitions=args.repetitions),
                 record("mixed_nested_input_result_axes",mixed=True,ta=True,tb=True,repetitions=args.repetitions)],
        "timing_scope":{
            "setup":"JIT owner construction plus fixture generation and independent oracle; not compiler-only cost",
            "first_public_reverse":"frontend/certificate, cold compilation, preparation, execution and copied gradients",
            "warm_public_reverse":"completed native_backward with compiler calls forbidden; includes frontend and native preparation/readback",
            "captured_members":"pure-SSA captured member device windows excluding capture/instantiation/copies",
            "interleaved_native_program":"128-repeat native program event windows including host dispatch gaps; input update and output copies excluded"},
        "claim":"owning public mapped f32 reverse through typed Graph/native AD/Schedule/Tile/ROCm/LLVM/checked ABI; no generic closure or primal f32 claim"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":main()
