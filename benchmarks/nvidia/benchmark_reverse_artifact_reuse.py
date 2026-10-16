"""Interleaved same-image comparison of public reverse adapter overhead on SM120."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera.compiler import native_attention_program
from tessera.compiler import native_attention_vjp_runtime as candidate
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_attention_vjp import clear_prepared, prepared
from benchmarks.nvidia.record_typed_attention_backward import (
    compare_results, decode, device, expected, source_snapshot,
)
from tests.device.nvidia.test_public_typed_attention_vjp import function


def record(args):
    if args.rounds < 3 or args.reps < 1:
        raise ValueError("require at least three interleaved rounds and positive repetitions")
    probe = device()
    before = source_snapshot()
    if before["status"]:
        raise RuntimeError("commit source before recording evidence")
    spec = importlib.util.spec_from_file_location(
        "tessera.compiler._reverse_adapter_control", args.control)
    if spec is None or spec.loader is None:
        raise ValueError("control must be an explicit Python adapter source")
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    execute = candidate.execute_family
    controls = {"parent": control.execute_family, "candidate": execute}
    folder = Path("benchmarks/baselines/sm120_saved_lse_half_backward_20261016/public-stage/benchmark")
    rows = []
    compile_native = native_attention_program.compile_attention_vjp_program
    compiled = {}
    def compile_once(graph, active, **kwargs):
        key = (graph, active, tuple(sorted((name, str(value)) for name, value in kwargs.items())))
        if key not in compiled:
            compiled[key] = compile_native(graph, active, **kwargs)
        return compiled[key]
    for storage in ("f16", "bf16", "f32"):
        for sq, sk in ((3, 5), (7, 129)):
            path = folder / f'{"f16" if storage == "f32" else storage}-sq{sq}-sk{sk}-bias1-lse1-inputs.npz'
            values = decode(path, "f16" if storage == "f32" else storage, storage == "f32")
            wrt = ("bias", "v", "k", "q")
            oracle = expected(values, True, True)
            fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=wrt)(function(True, True, True))
            try:
                with ExitStack() as stack:
                    host = (*tuple(values[name] for name in fn.arg_names), values["do"], values["seed"])
                    resident = tuple(stack.enter_context(NvidiaDeviceSession()).upload(value) for value in host)
                    for lane, roots in (("host", host), ("resident", resident)):
                        identity, handles, snapshots = {}, {}, []
                        samples = {arm: {name: [] for name in ("completed_ms", "forward_event_ms", "backward_event_ms")}
                                   for arm in controls}
                        for arm, adapter in controls.items():
                            # Both adapters receive the identical real compiler product.
                            # Cold/warm receipt fields otherwise alter package identity.
                            with patch.object(native_attention_program, "compile_attention_vjp_program", compile_once), patch.object(candidate, "execute_family", adapter):
                                for _ in range(3):
                                    result = fn.native_backward(*roots[:-2], out_cotangents=roots[-2:])
                                compare_results(result, oracle, wrt)
                                artifact = fn.native_backward_runtime_artifact()
                                owner = prepared(artifact.metadata)
                                identity[arm] = (artifact.artifact_hash, artifact.metadata["program_digest"])
                                handles[arm] = owner.handle
                                snapshots.append((result, tuple(x.copy() for x in result)))
                        assert identity["parent"] == identity["candidate"]
                        assert handles["parent"] == handles["candidate"]
                        def forbidden(*unused, **kwargs):
                            raise AssertionError("timed public reverse invoked compiler or concrete frontend")
                        with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden), patch("subprocess.check_output", forbidden), patch.object(fn, "_fn", forbidden):
                            for trial in range(args.rounds):
                                # ABBA/BAAB windows balance order and drift. Preserve raw samples.
                                order = ("parent", "candidate", "candidate", "parent")
                                if trial % 2:
                                    order = ("candidate", "parent", "parent", "candidate")
                                for arm in order:
                                    with patch.object(candidate, "execute_family", controls[arm]):
                                        for _ in range(args.reps):
                                            start = time.perf_counter()
                                            result = fn.native_backward(*roots[:-2], out_cotangents=roots[-2:])
                                            elapsed = (time.perf_counter() - start) * 1e3
                                            owner = prepared(fn.native_backward_runtime_artifact().metadata)
                                            assert owner.handle == handles[arm]
                                            events = owner.last_device_ms
                                            assert events and all(np.isfinite(x) and x > 0 for x in events)
                                            samples[arm]["completed_ms"].append(elapsed)
                                            samples[arm]["forward_event_ms"].append(events[0])
                                            samples[arm]["backward_event_ms"].append(events[1])
                                    compare_results(result, oracle, wrt)
                        for original, held in snapshots:
                            for actual, snapshot in zip(original, held, strict=True):
                                np.testing.assert_array_equal(actual, snapshot)
                        medians = {arm: {name: statistics.median(data) for name, data in sample.items()}
                                   for arm, sample in samples.items()}
                        rows.append(dict(storage=storage, sq=sq, sk=sk, lane=lane,
                            input_file=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            artifact_hash=identity["candidate"][0], program_digest=identity["candidate"][1],
                            shared_native_owner=True, samples=samples, medians=medians,
                            completed_call_parent_over_candidate=medians["parent"]["completed_ms"]/medians["candidate"]["completed_ms"]))
                        print(storage, sq, sk, lane, rows[-1]["completed_call_parent_over_candidate"], flush=True)
            finally:
                fn.close_native_storage()
                clear_prepared()
    after = source_snapshot()
    assert before == after
    packet = dict(schema="tessera.sm120_reverse_adapter_ab.v1", source_before=before, source_after=after,
                  native_programs_compiled=len(compiled),
                  control_sha256=hashlib.sha256(args.control.read_bytes()).hexdigest(),
                  control_source=str(args.control), device=probe, rounds=args.rounds, reps=args.reps,
                  correctness="independent_fp64_before_and_after_each_window; retained_outputs_unchanged",
                  compiler_and_concrete_frontend_forbidden_during_timing=True,
                  scope="warm public completed calls; unchanged native images and shared owner; no kernel speedup claim",
                  rows=rows)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--reps", type=int, default=20)
    record(parser.parse_args())
