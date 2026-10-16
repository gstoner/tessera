"""Exact SM120 saved-LSE storage: correctness, kernel events, completed calls, replay."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from tests.device.nvidia.test_saved_lse_storage import (
    saved, saved_open, saved_bias, saved_bias_open, reference, compare,
)


def inputs(row):
    storage = np.dtype(ml_dtypes.bfloat16 if row["storage"] == "bfloat16" else row["storage"])
    b, hq, hkv, sq, sk, d, dv = row["shape"]
    rng = np.random.default_rng(20261016)
    values = {name: rng.normal(0, .2, shape).astype(storage) for name, shape in {
        "q": (b, hq, sq, d), "k": (b, hkv, sk, d), "v": (b, hkv, sk, dv)}.items()}
    if row["bias"]:
        values["bias"] = rng.normal(0, .1, (b, hq, 1, 1)).astype(np.float32)
    return values, storage


def source():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    tools = {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in (
        "TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")}
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "native_source_tree": git("rev-parse", "HEAD:src"), "tools": tools}


def record(row, repetitions, directory, index):
    values, storage = inputs(row)
    expected = reference(values, row["causal"])
    function = (saved_bias if row["causal"] else saved_bias_open) if row["bias"] else (
        saved if row["causal"] else saved_open)
    fn = ts.jit(target="nvidia_sm120")(function)
    wall = {"host": [], "resident": []}
    events = {"host": [], "resident": []}
    with ExitStack() as stack:
        stack.callback(fn.close_native_storage)
        sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in values}
        roots = {name: Borrowed(sessions[name].upload(value)) for name, value in values.items()}
        for session in sessions.values():
            assert session.synchronize() == 0
        def call(mode):
            return fn(**(roots if mode == "resident" else values))
        first = call("host")
        assert fn._native_prepared_attention_calls
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        compare(first, expected, storage)
        compare(call("resident"), expected, storage)
        handle = owner.handle
        for iteration in range(repetitions):
            for mode in (("host", "resident") if iteration % 2 else ("resident", "host")):
                start = time.perf_counter()
                pair = call(mode)
                wall[mode].append((time.perf_counter()-start)*1000)
                events[mode].append(owner.last_device_ms)
                assert owner.handle == handle
                compare(pair, expected, storage)
                for actual, original in zip(pair, first, strict=True):
                    np.testing.assert_array_equal(actual, original)
        artifact_path = f"packages/{index:02d}.json.gz"
        (directory / artifact_path).write_bytes(gzip.compress(owner.artifact.to_json().encode(), mtime=0))
        descriptor = owner.descriptor
        in_bindings = sorted((binding for binding in descriptor.buffers if binding.direction == "input"),
                             key=lambda binding: binding.ordinal)
        row.update(
            artifact=artifact_path,
            input_bindings={binding.name: fn.arg_names[position]
                            for binding, position in zip(in_bindings, owner.positions, strict=True)},
            abi=descriptor.abi_id, entry=descriptor.entry_symbol,
            image_digest=owner.image.image_digest, descriptor_digest=descriptor.descriptor_digest,
            provenance=descriptor.provenance,
            max_abs_error=[float(np.max(np.abs(a.astype(np.float64)-e)))
                           for a, e in zip(first, expected, strict=True)],
            public_completed_call_samples_ms=wall,
            public_completed_call_medians_ms={name: statistics.median(items) for name, items in wall.items()},
            native_kernel_event_samples_ms=events,
            native_kernel_event_medians_ms={name: statistics.median(items) for name, items in events.items()},
            stages=[{"stage": name, "producer": stage.producer, "input_digest": stage.input_digest,
                     "output_digest": stage.output_digest} for name, stage in (
                ("graph", owner.compiled.bundle.graph), ("schedule", owner.compiled.bundle.schedule),
                ("tile", owner.compiled.bundle.tile), ("target", owner.compiled.bundle.target_ir),
                ("backend", owner.compiled.bundle.backend))],
        )
    return row


def replay(directory):
    packet = json.loads((directory / "packet.json").read_text())
    results = []
    for row in packet["rows"]:
        values, storage = inputs(row)
        expected = reference(values, row["causal"])
        artifact = rt.RuntimeArtifact.from_json(gzip.decompress(
            (directory / row["artifact"]).read_bytes()).decode())
        b, hq, _, sq, _, _, dv = row["shape"]
        outputs = (np.zeros((b, hq, sq, dv), storage), np.zeros((b, hq, sq), np.float32))
        arguments = {name: values[source] for name, source in row["input_bindings"].items()}
        bindings = sorted((binding for binding in artifact.launch_descriptor.buffers
                           if binding.direction == "output"), key=lambda binding: binding.ordinal)
        arguments.update((binding.name, output) for binding, output in zip(bindings, outputs, strict=True))
        arguments.update(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), row["shape"], strict=True))
        if row["bias"]:
            arguments.update(zip(("BiasB", "BiasH", "BiasQ", "BiasK"), values["bias"].shape, strict=True))
        result = rt.launch(artifact, arguments)
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        compare(outputs, expected, storage)
        results.append({"artifact": row["artifact"], "execution_kind": result["execution_kind"]})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("requires owning RTX5070/SM120")
    if args.replay_only:
        def forbidden(*a, **k):
            raise RuntimeError("compiler or subprocess invoked during serialized replay")
        subprocess.run = forbidden
        subprocess.check_output = forbidden
        results = replay(args.output)
        (args.output / "replay.json").write_text(json.dumps(
            {"device": gpu, "compiler_subprocesses_forbidden": True, "rows": results}, indent=2)+"\n")
        print(f"compiler-free numerical replay: {len(results)} passed")
        return
    if args.repetitions < 3:
        raise ValueError("requires at least three alternating rounds")
    before = source()
    if before["status"]:
        raise RuntimeError("recording requires a clean source freeze")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "packages").mkdir(exist_ok=True)
    rows = [{"storage": storage, "shape": list(shape), "causal": causal, "bias": bias}
        for storage in ("float16", "bfloat16", "float32")
        for shape in ((1, 2, 1, 3, 5, 4, 3), (1, 2, 1, 32, 129, 32, 32))
        for causal in (False, True) for bias in (False, True)]
    for index, row in enumerate(rows):
        record(row, args.repetitions, args.output, index)
    after = source()
    assert before == after
    packet = {"device": gpu, "architecture": "sm120", "source_before": before, "source_after": after,
        "rows": rows, "correctness": "independent FP64 oracle from exact stored operands before timing",
        "timing_scope": "CUDA events cover kernel only; completed public calls include waits, snapshots and host O/LSE downloads",
        "scope": "static compact homogeneous QKV; optional broadcast FP32 bias; O matches input; LSE FP32; no AD or default promotion"}
    (args.output / "packet.json").write_text(json.dumps(packet, indent=2)+"\n")
    print(f"recorded {len(rows)} correctness-gated rows")


if __name__ == "__main__":
    main()
