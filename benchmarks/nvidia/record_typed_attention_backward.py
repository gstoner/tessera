"""Matched stored-value typed public VJP characterization on the owning SM120."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_attention_vjp import prepared, clear_prepared
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference, compare


def source_snapshot():
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
        "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD:src"], text=True).strip(),
        "tools": {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                  for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def device():
    probe = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"], text=True).strip()
    if rt._nvidia_device_name() != "sm_120" or "RTX 5070" not in probe or probe.split(",")[2].strip() != "12.0":
        raise RuntimeError("typed backward recorder requires the owning SM120 GPU")
    # Complete host/library discovery before the compiler-subprocess guard.
    # No package is prepared or launched by this capability probe.
    rt.backend_capabilities("nvidia_sm120")
    return probe


def decode(path, storage, control):
    if storage not in ("f16", "bf16") or type(control) is not bool:
        raise ValueError("typed recorder requires explicit f16/bf16 source and boolean control")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bf16" else "float16")
    with np.load(path, allow_pickle=False) as values:
        for name in ("q", "k", "v", "do"):
            if values[name].dtype != np.dtype("<u2"):
                raise ValueError("typed recorder source requires little-endian raw u16 words")
        if any(values[name].dtype != np.dtype("<f4") for name in ("bias", "seed")):
            raise ValueError("typed recorder auxiliaries require f32 storage")
        result = {name: values[name].view(dtype).copy() for name in ("q", "k", "v", "do")}
        result.update(bias=values["bias"].copy(), seed=values["seed"].copy())
    if control:
        for name in ("q", "k", "v", "do"):
            result[name] = result[name].astype(np.float32)
    return result


def expected(values, biased, causal):
    _, gradients = reference(values["q"], values["k"], values["v"], values["do"],
                             values["bias"] if biased else None, values["seed"], causal=causal)
    return dict(zip(("q", "k", "v", "bias")[:len(gradients)], gradients, strict=True))


def compare_results(results, oracle, wrt):
    for name, result in zip(wrt, results, strict=True):
        compare(result, oracle[name])


def run_case(folder, label, datafile, storage, control, dims, biased, saved, causal, rounds, reps):
    values = decode(folder/datafile, storage, control)
    wrt = ("bias", "v", "k", "q") if biased else ("v", "k", "q")
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=wrt)(function(saved, biased, causal))
    oracle = expected(values, biased, causal)
    roots = {name: values[name] for name in (("bias", "v", "q", "k") if biased else ("q", "k", "v"))}
    seeds = (values["do"], values["seed"]) if saved else values["do"]
    output_rows = {}
    retained = []
    try:
        with ExitStack() as stack:
            host_args = tuple(roots[name] for name in fn.arg_names)
            resident_inputs = tuple(stack.enter_context(NvidiaDeviceSession()).upload(value)
                                    for value in (*host_args, values["do"], *((values["seed"],) if saved else ())))
            resident_args = resident_inputs[:len(host_args)]
            resident_seeds = (resident_inputs[len(host_args)], resident_inputs[-1]) if saved else resident_inputs[-1]
            calls = {"host": lambda: fn.native_backward(*host_args, out_cotangents=seeds),
                     "resident": lambda: fn.native_backward(*resident_args, out_cotangents=resident_seeds)}
            artifacts = {}
            owners = {}
            for lane, call in calls.items():
                start = time.perf_counter()
                result = call()
                cold_ms = (time.perf_counter()-start)*1e3
                compare_results(result, oracle, wrt)
                retained.append((result, tuple(x.copy() for x in result)))
                artifact = fn.native_backward_runtime_artifact()
                artifacts[lane] = artifact
                owners[lane] = prepared(artifact.metadata)
                path = label+"-"+lane+".json.gz"
                (folder/path).write_bytes(gzip.compress(artifact.to_json().encode(), mtime=0))
                output_rows[lane] = dict(artifact=path, artifact_hash=artifact.artifact_hash,
                    program_digest=artifact.metadata["program_digest"], owner_handle=owners[lane].handle,
                    cold_compile_prepare_launch_ms=cold_ms,
                    max_abs_error={name: float(np.max(np.abs(value.astype(np.float64)-oracle[name])))
                                   for name, value in zip(wrt, result, strict=True)},
                    public_completed_call_samples_ms=[],
                    native_forward_event_samples_ms=[], native_backward_event_samples_ms=[])
            def forbidden(*args, **kwargs):
                raise AssertionError("timed/replayed typed VJP invoked compiler or concrete frontend")
            with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden), patch("subprocess.check_output", forbidden), patch.object(fn, "_fn", forbidden):
                for trial in range(rounds):
                    order = ("host", "resident") if trial % 2 == 0 else ("resident", "host")
                    for lane in order:
                        wall, forward, backward = [], [], []
                        for _ in range(reps):
                            start = time.perf_counter()
                            result = calls[lane]()
                            wall.append((time.perf_counter()-start)*1e3)
                            owner = prepared(fn.native_backward_runtime_artifact().metadata)
                            assert owner.handle == owners[lane].handle
                            assert owner.last_device_ms and all(x > 0 for x in owner.last_device_ms)
                            forward.append(owner.last_device_ms[0]); backward.append(owner.last_device_ms[1])
                        compare_results(result, oracle, wrt)
                        row = output_rows[lane]
                        for name, value in zip(wrt, result, strict=True):
                            row["max_abs_error"][name] = max(row["max_abs_error"][name], float(np.max(np.abs(value.astype(np.float64)-oracle[name]))))
                        row["public_completed_call_samples_ms"].append(statistics.median(wall))
                        row["native_forward_event_samples_ms"].append(statistics.median(forward))
                        row["native_backward_event_samples_ms"].append(statistics.median(backward))
                replay = rt.launch(artifacts["host"], (*host_args, values["do"], *((values["seed"],) if saved else ())))
                assert replay["ok"] and replay["execution_kind"] == "native_gpu", replay
                compare_results(replay["output"], oracle, wrt)
            for original, snapshot in retained:
                for actual, previous in zip(original, snapshot, strict=True):
                    np.testing.assert_array_equal(actual, previous)
            for row in output_rows.values():
                for prefix in ("public_completed_call", "native_forward_event", "native_backward_event"):
                    row[prefix+"_median_ms"] = statistics.median(row[prefix+"_samples_ms"])
            return dict(case=label, input_file=datafile,
                input_sha256=hashlib.sha256((folder/datafile).read_bytes()).hexdigest(), source_storage=storage,
                execution_storage="fp32_matched_quantized_control" if control else storage,
                dims=list(dims), bias=biased, saved_lse=saved, causal=causal, wrt=list(wrt),
                correctness="independent_fp64_before_after_trials_and_portable_replay",
                compiler_subprocesses="forbidden_during_timing_and_replay",
                host_resident_program_identical=output_rows["host"]["program_digest"] == output_rows["resident"]["program_digest"],
                lanes=output_rows)
    finally:
        fn.close_native_storage()
        clear_prepared()


def record(args):
    probe = device()
    before = source_snapshot()
    if before["status"] and not args.allow_dirty_diagnostic:
        raise RuntimeError("benchmark requires a clean committed source")
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = itertools.product(("f16", "bf16"), ((1, 2, 1, 3, 5, 4, 3), (1, 4, 2, 7, 129, 8, 6)), (False, True), (False, True))
    for index, (storage, dims, biased, saved) in enumerate(cases):
        if args.case_limit and index >= args.case_limit:
            break
        b, hq, hkv, sq, sk, d, dv = dims
        dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bf16" else "float16")
        rng = np.random.default_rng(20261016)
        arrays = {name: rng.normal(0, .3, shape).astype(dtype).view(np.uint16) for name, shape in {
            "q": (b, hq, sq, d), "k": (b, hkv, sk, d), "v": (b, hkv, sk, dv), "do": (b, hq, sq, dv)}.items()}
        arrays["bias"] = rng.normal(0, .1, (1, hq, 1, sk)).astype(np.float32)
        arrays["seed"] = rng.normal(0, .2, (b, hq, sq)).astype(np.float32) if saved else np.zeros((b, hq, sq), np.float32)
        stem = f"{storage}-sq{sq}-sk{sk}-bias{int(biased)}-lse{int(saved)}"
        datafile = stem+"-inputs.npz"
        np.savez_compressed(folder/datafile, **arrays)
        for control in ((False, True) if index % 2 == 0 else (True, False)):
            label = stem+("-fp32-control" if control else "-native")
            row = run_case(folder, label, datafile, storage, control, dims, biased, saved, True, args.rounds, args.reps)
            rows.append(row)
            print(json.dumps({"case": label, "host_ms": row["lanes"]["host"]["public_completed_call_median_ms"],
                              "resident_ms": row["lanes"]["resident"]["public_completed_call_median_ms"]}), flush=True)
    after = source_snapshot()
    assert before == after
    result = dict(schema="tessera.sm120.typed_attention_vjp_characterization.v1", source_before=before,
        source_after=after, device=probe, diagnostic=bool(args.allow_dirty_diagnostic or args.case_limit or args.rounds < 7),
        rounds=args.rounds, repetitions=args.reps, rows=rows,
        input_encoding="source-format raw u16 words; FP32 controls expand identical quantized values",
        event_scope="native CUDA event windows around forward/backward; excludes snapshots/downloads and may include dispatch gaps",
        wall_scope="completed public native_backward including binding, snapshots, allocation and returned host gradients",
        comparison_scope="storage/residual-policy characterization; no default or general speedup promotion")
    (folder/"index.json").write_text(json.dumps(result, indent=2)+"\n")


def replay(index):
    probe = device()
    document = json.loads(index.read_text())
    rows = []
    for row in document["rows"]:
        assert hashlib.sha256((index.parent/row["input_file"]).read_bytes()).hexdigest() == row["input_sha256"]
        values = decode(index.parent/row["input_file"], row["source_storage"], row["execution_storage"] == "fp32_matched_quantized_control")
        oracle = expected(values, row["bias"], row["causal"])
        raw = gzip.decompress((index.parent/row["lanes"]["host"]["artifact"]).read_bytes()).decode()
        artifact = rt.RuntimeArtifact.from_json(raw)
        assert artifact.artifact_hash == row["lanes"]["host"]["artifact_hash"]
        order = ("bias", "v", "q", "k") if row["bias"] else ("q", "k", "v")
        inputs = tuple(values[name] for name in order) + (values["do"],) + ((values["seed"],) if row["saved_lse"] else ())
        def forbidden(*args, **kwargs):
            raise AssertionError("fresh portable VJP replay invoked a compiler subprocess")
        with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden), patch("subprocess.check_output", forbidden):
            receipt = rt.launch(artifact, inputs)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        compare_results(receipt["output"], oracle, row["wrt"])
        rows.append(dict(case=row["case"], artifact_hash=artifact.artifact_hash, execution_kind=receipt["execution_kind"]))
        clear_prepared()
    (index.parent/"fresh-replay.json").write_text(json.dumps(dict(device=probe, compiler_subprocesses_forbidden=True,
        preflight="device and host/library capability discovery only; no package prepared or launched",
        first_package_preparation_inside_subprocess_guard=True, rows=rows), indent=2)+"\n")
    print(json.dumps({"fresh_replayed": len(rows), "device": probe}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.reps < 1 or args.case_limit < 0:
        parser.error("positive rounds/repetitions and nonnegative case limit required")
    if args.replay:
        replay(args.replay)
    elif args.output:
        record(args)
    else:
        parser.error("provide --output or --replay")


if __name__ == "__main__":
    main()
