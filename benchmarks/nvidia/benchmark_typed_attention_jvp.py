"""Correctness-gated prepared typed JVP characterization; no default promotion."""
import argparse
import hashlib
import json
import lzma
import math
import shutil
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.native_attention_jvp_runtime import prepared
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

def record(output: Path, repetitions: int = 9):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be positive")
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or not device.endswith(", 12.0"):
        raise RuntimeError("exact single SM120 owning device required")
    rows=[]
    for sq,sk in [(3,5),(17,129)]:
        for half in ["float16","bfloat16"]:
            half_dtype=np.dtype(ml_dtypes.bfloat16 if half=="bfloat16" else half)
            rng=np.random.default_rng(20261017)
            common={n:rng.normal(0,.3,shape).astype(half_dtype) for n,shape in {
                "q":(1,2,sq,16),"k":(1,1,sk,16),"v":(1,1,sk,16)}.items()}
            directions={n:rng.normal(0,.1,x.shape).astype(half_dtype) for n,x in common.items()}
            for wide in [False,True]:
                values={n:x.astype(np.float32) if wide else x for n,x in common.items()}
                seeds=tuple(x.astype(np.float32) if wide else x for x in directions.values())
                fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=("q","k","v"))(function(True,False,True))
                try:
                    expected=reference(*values.values(),*seeds)
                    actual=fn.native_jvp(**values,tangents=seeds)
                    for got,wanted in zip(actual,expected,strict=True):
                        for a,b in zip(got,wanted,strict=True):compare(a,b)
                    package=next(iter(fn._native_jvp_packages.values()))
                    metadata=package.contract["steps"][0]["child_metadata"]
                    owner=prepared(metadata)
                    artifact_path=output.parent/"artifacts"/(metadata["program_digest"]+".json.xz")
                    artifact_path.parent.mkdir(parents=True,exist_ok=True)
                    artifact_bytes=lzma.compress(metadata["program_json"].encode())
                    artifact_path.write_bytes(artifact_bytes)
                    for _ in range(3):fn.native_jvp(**values,tangents=seeds)
                    e2e=[];device=[]
                    for _ in range(repetitions):
                        start=time.perf_counter()
                        last=fn.native_jvp(**values,tangents=seeds)
                        e2e.append(1000*(time.perf_counter()-start))
                        device.append(list(owner.last_device_ms))
                    for got,wanted in zip(last,expected,strict=True):
                        for a,b in zip(got,wanted,strict=True):compare(a,b)
                    row=dict(shape=[1,2,1,sq,sk,16,16],source_quantization=half,
                        storage="float32" if wide else half,saved_lse=True,
                        correctness="independent_fp64_before_timing",
                        program_digest=metadata["program_digest"],
                        program_artifact=str(artifact_path.relative_to(output.parent)),
                        program_artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                        forward_image_sha256=hashlib.sha256(owner.program.pair.forward.image.payload).hexdigest(),
                        tangent_image_sha256=hashlib.sha256(owner.program.tangent.image).hexdigest(),
                        host_sizer_sha256=hashlib.sha256(owner.program.tangent.host_library).hexdigest(),
                        forward_entry=owner.program.pair.forward.descriptor.entry_symbol,
                        tangent_entry=owner.program.tangent.entry,
                        completed_public_host_samples_ms=e2e,completed_public_host_median_ms=median(e2e),
                        native_forward_jvp_event_samples_ms=device,
                        native_forward_event_median_ms=median(x[0] for x in device),
                        native_jvp_event_median_ms=median(x[1] for x in device))
                    rows.append(row);print(json.dumps(row),flush=True)
                finally:fn.close_native_storage()
    root=Path.cwd()
    paths=[
        "benchmarks/nvidia/benchmark_typed_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_runtime.py",
        "python/tessera/compiler/native_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_artifact.py",
        "python/tessera/compiler/native_attention_program.py",
        "python/tessera/compiler/native_gpu_tensor.py",
        "python/tessera/compiler/native_jvp_plugins.py",
        "python/tessera/compiler/jit.py",
        "python/tessera/compiler/resident_attention.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/programming_model/lib/NativeAttentionJvp.h",
        "src/compiler/ir/TangentInterface.cpp",
        "src/compiler/ir/include/Tessera/IR/AttentionTangentZero.h",
        "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
        "src/transforms/lib/AutodiffForwardPass.cpp",
    ]
    result=dict(device=subprocess.check_output(["nvidia-smi","--query-gpu=name,compute_cap","--format=csv,noheader"],text=True).strip(),
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True)),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        scope="static saved-LSE, causal, GQA2, no bias; quantized-common-value f32 controls; compile, initial preparation and oracle excluded; public calls include validation/upload/readback; CUDA event dispatch windows reported separately, not hardware-counter isolated kernel proof; no default or speedup promotion",
        rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def _paired_options(windows, min_window_ms):
    if type(windows) is not int or windows < 4 or windows % 2:
        raise ValueError("paired windows require an even count of at least four")
    if (type(min_window_ms) not in (int, float) or
            not math.isfinite(min_window_ms) or min_window_ms < 10):
        raise ValueError("completed-call window floor must be finite and at least 10 ms")


def _completed_window(launch, owner, repetitions, clock):
    calls, events = [], []
    last = None
    for _ in range(repetitions):
        begin = clock()
        last = launch()
        elapsed = 1000 * (clock() - begin)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("completed public call has no positive finite duration")
        native = owner.last_device_ms
        if (native is None or len(native) != 2 or
                any(not math.isfinite(x) or x < 0 for x in native)):
            raise RuntimeError("native forward/JVP event samples are unavailable")
        calls.append(elapsed)
        events.append(list(native))
    return dict(repetitions=repetitions, completed_call_samples_ms=calls,
                completed_call_sum_ms=sum(calls), native_event_samples_ms=events,
                last=last)


def _paired_completed_windows(arms, validate, windows=10, min_window_ms=20,
                              clock=time.perf_counter):
    """Counterbalanced complete calls; validation and sample reads are untimed."""
    _paired_options(windows, min_window_ms)
    if len(arms) != 2:
        raise ValueError("paired timing requires exactly two arms")
    names = list(arms)
    accepted, rejected = [], []
    repetitions = 1
    while len(accepted) < windows:
        order = names if len(accepted) % 2 == 0 else list(reversed(names))
        pair = {}
        for name in order:
            launch, owner = arms[name]
            validate(launch())
            row = _completed_window(launch, owner, repetitions, clock)
            validate(row.pop("last"))
            pair[name] = row
        record = dict(order=list(order), arms=pair)
        if min(row["completed_call_sum_ms"] for row in pair.values()) < min_window_ms:
            rejected.append(record)
            repetitions *= 2
            if repetitions > 65536:
                raise RuntimeError("completed-call window could not reach its floor")
            continue
        accepted.append(record)
    summaries = {}
    for name in names:
        spans = [row["arms"][name] for row in accepted]
        summaries[name] = dict(
            completed_public_host_median_ms=median(
                x["completed_call_sum_ms"]/x["repetitions"] for x in spans),
            native_forward_event_median_ms=median(
                sample[0] for x in spans for sample in x["native_event_samples_ms"]),
            native_jvp_event_median_ms=median(
                sample[1] for x in spans for sample in x["native_event_samples_ms"]))
    return dict(min_completed_call_window_ms=min_window_ms, windows=accepted,
                rejected_calibration_windows=rejected, summary=summaries)


def _archive_owner(owner, directory):
    digest, raw = owner.identity
    encoded = lzma.compress(raw.encode())
    path = directory/"artifacts"/(digest+".json.xz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("an existing program archive differs from its immutable pin")
    path.write_bytes(encoded)
    return dict(program_digest=digest, artifact=str(path.relative_to(directory)),
                artifact_sha256=hashlib.sha256(encoded).hexdigest(),
                forward_image_sha256=hashlib.sha256(owner.program.pair.forward.image.payload).hexdigest(),
                tangent_image_sha256=hashlib.sha256(owner.program.tangent.image).hexdigest(),
                host_sizer_sha256=hashlib.sha256(owner.program.tangent.host_library).hexdigest())


def _bounded_timing_source_snapshot():
    root = Path.cwd()
    paths = [
        "benchmarks/nvidia/benchmark_typed_attention_jvp.py",
        "python/tessera/compiler/native_bounded_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_runtime.py",
        "python/tessera/compiler/native_attention_program.py",
        "python/tessera/compiler/native_attention_jvp_artifact.py",
        "python/tessera/compiler/native_attention_jvp.py",
        "python/tessera/compiler/frontend_authority.py",
        "python/tessera/compiler/jit.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "tests/device/nvidia/test_public_typed_attention_jvp.py",
        "tests/device/nvidia/test_public_typed_attention_vjp.py",
        "tests/device/nvidia/test_saved_lse_half_backward_kernel.py",
    ]
    return dict(
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True)),
        source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        runtime_sha256=hashlib.sha256(
            Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        target_compiler_sha256=(hashlib.sha256(
            Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest()
            if os.environ.get("TESSERA_NVIDIA_OPT") else None))


def record_bounded(output, windows=10, min_window_ms=20, reverse=False):
    """Compare bounded and static public owners on identical typed operands."""
    _paired_options(windows, min_window_ms)
    output = Path(output)
    if output.exists():
        raise ValueError("choose a fresh timing output to preserve existing evidence")
    before = _bounded_timing_source_snapshot()
    if before["source_dirty"]:
        raise ValueError("commit the timing source before collecting evidence")
    from contextlib import ExitStack
    from unittest.mock import patch
    from tessera import runtime as rt
    from tessera.compiler.native_attention_jvp_runtime import clear_prepared
    executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    device = subprocess.check_output([executable,
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
        text=True).strip()
    if (len(device.splitlines()) != 1 or "RTX 5070" not in device or
            device.split(",")[2].strip() != "12.0" or rt._nvidia_device_name() != "sm_120"):
        raise RuntimeError("owning RTX5070 / SM120 required")
    shapes = [(1, 1), (7, 9), (31, 65), (127, 131)]
    if reverse:
        shapes.reverse()
    rows = []
    def forbidden(*args, **kwargs):
        raise AssertionError("timed replay attempted compilation or concrete frontend evaluation")
    for storage in ("float16", "bfloat16"):
        dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
        for bias in (False, True):
            active = ("q", "k", "v", "bias") if bias else ("q", "k", "v")
            bounded = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
                function(True, bias, True))
            static = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
                function(True, bias, True))
            bounded_handle = None
            try:
                for sq, sk in shapes:
                    rng = np.random.default_rng(20261017 + sq*131 + sk)
                    values = {n: rng.normal(0, .3, shape).astype(dtype) for n, shape in {
                        "q": (1, 2, sq, 16), "k": (1, 1, sk, 16),
                        "v": (1, 1, sk, 16)}.items()}
                    if bias:
                        values["bias"] = rng.normal(0, .1, (1, 2, 1, 1)).astype(np.float32)
                    directions = {n: rng.normal(0, .1, x.shape).astype(x.dtype)
                                  for n, x in values.items()}
                    seeds = tuple(directions.values())
                    expected = reference(*(values[n] for n in ("q", "k", "v")),
                        *(directions[n] for n in ("q", "k", "v")), causal=True,
                        bias=values.get("bias"), dbias=directions.get("bias"))
                    def validate(result):
                        for actuals, targets in zip(result, expected, strict=True):
                            for actual, target in zip(actuals, targets, strict=True):
                                compare(actual, target)
                    def launch(fn, is_bounded):
                        result = fn.native_jvp(**values, tangents=seeds,
                            **({"sequence_bounds": (127, 131)} if is_bounded else {}))
                        if (not fn.last_jvp_execution or
                                fn.last_jvp_execution.get("execution_kind") != "native_gpu"):
                            raise RuntimeError("timing arm did not execute its native GPU route")
                        return result
                    for _ in range(3):
                        validate(launch(bounded, True))
                        validate(launch(static, False))
                    call = next(iter(bounded._native_prepared_jvp_calls.values()))
                    bounded_owner = call.owner
                    if bounded_handle is None:
                        bounded_handle = bounded_owner.handle
                    if bounded_owner.handle != bounded_handle or len(bounded._native_prepared_jvp_calls) != 1:
                        raise RuntimeError("bounded public call did not retain one capacity owner")
                    package = next(reversed(static._native_jvp_packages.values()))
                    metadata = package.contract["steps"][0]["child_metadata"]
                    static_owner = prepared(metadata)
                    identities = {"bounded": _archive_owner(bounded_owner, output.parent),
                                  "static": _archive_owner(static_owner, output.parent)}
                    arms = {"bounded": (lambda: launch(bounded, True), bounded_owner),
                            "static": (lambda: launch(static, False), static_owner)}
                    if reverse:
                        arms = dict(reversed(list(arms.items())))
                    with ExitStack() as guard:
                        guard.enter_context(patch("subprocess.run", forbidden))
                        for fn in (bounded, static):
                            guard.enter_context(patch.object(fn, "_fn", forbidden))
                            guard.enter_context(patch.object(fn, "_trace_frontend_capture", forbidden))
                        measured = _paired_completed_windows(
                            arms, validate, windows=windows, min_window_ms=min_window_ms)
                    summaries = measured["summary"]
                    row = dict(storage=storage, bias=bias, saved_lse=True,
                        actual_sq_sk=[sq, sk], capacities_sq_sk=[127, 131],
                        correctness="independent_fp64_before_and_after_each_window",
                        same_bounded_owner_across_shapes=True, identities=identities,
                        paired_completed_host=measured,
                        completed_host_static_over_bounded=(
                            summaries["static"]["completed_public_host_median_ms"] /
                            summaries["bounded"]["completed_public_host_median_ms"]))
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("storage", "bias", "actual_sq_sk",
                        "completed_host_static_over_bounded")}), flush=True)
            finally:
                bounded.close_native_storage()
                static.close_native_storage()
                clear_prepared()
    after = _bounded_timing_source_snapshot()
    if after != before:
        raise RuntimeError("timing source or selected tools changed during measurement")
    result = dict(schema="tessera.sm120.bounded_prepared_jvp_timing.v1",
        architecture="sm_120", device=device, **before,
        reverse=reverse, rows=rows,
        scope="causal GQA2 saved-LSE public host calls; compile/preparation/oracle excluded; "
              "CUDA event dispatch and completed public-call intervals are separate; "
              "paired arms use identical typed values; explicit bounds only; no default promotion")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9)
    parser.add_argument("--bounded",action="store_true")
    parser.add_argument("--windows",type=int,default=10)
    parser.add_argument("--min-window-ms",type=float,default=20)
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    if args.bounded:
        record_bounded(args.output,args.windows,args.min_window_ms,args.reverse)
    else:
        record(args.output,args.repetitions)


if __name__=="__main__":
    main()
