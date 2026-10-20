"""Paired bounded/static RMSNorm raw-member timing on owning SM120."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.unit.test_sm120_bounded_norm_schedule import source
from tests.unit.test_sm120_reverse_row_schedule import invoke


def compile_norm(tool, storage, capacity, mode, *, bounded):
    graph = source(storage, capacity).replace('numeric_policy = {accum = "fp32"}',
                                              f'numeric_policy = {{accum = "fp32"}}, schedule = "{mode}"')
    if not bounded:
        graph = re.sub(r' attributes \{\s*tessera.native.query_sequence_bound = \d+ : i64\}', "", graph)
        graph = graph.replace("x?x", f"x{capacity[2]}x")
    started = time.perf_counter()
    stages = [graph]
    for flag in ("--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
                 "--lower-tile-to-nvidia=sm=120", "--lower-tessera-nvidia-to-nvvm"):
        result = invoke(tool, stages[-1], flag)
        if result.returncode:
            raise RuntimeError(result.stderr)
        stages.append(result.stdout)
    entry = re.search(r'\bllvm.func @(\w+)\(', stages[2])[1]
    lowered, ptx, resources, compiler, toolchain, *_ = _compile_tile_ir(stages[2], entry)
    receipt = {"entry": entry, "compile_host_ms": (time.perf_counter() - started) * 1e3,
               "compiler": compiler, "toolchain": toolchain, "packager_resources": dict(resources),
               "stage_sha256": {key: hashlib.sha256(text.encode()).hexdigest()
                                for key, text in zip(("graph", "schedule", "tile", "target", "native", "lowered", "ptx"),
                                                     (*stages, lowered, ptx), strict=True)}}
    return ptx, receipt


def function_resources(lib, function):
    query = lib.cuFuncGetAttribute
    query.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p]
    query.restype = ct.c_int
    record = {}
    for name, attribute in (("max_threads_per_block", 0), ("shared_bytes", 1),
                            ("local_bytes", 3), ("registers", 4)):
        result = ct.c_int()
        if query(ct.byref(result), attribute, function):
            raise RuntimeError("native function resource query failed")
        record[name] = result.value
    occupancy = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t]
    occupancy.restype = ct.c_int
    active = ct.c_int()
    if occupancy(ct.byref(active), function, 128, 0):
        raise RuntimeError("native occupancy query failed")
    record.update(launch_threads=128, dynamic_shared_bytes=0, active_blocks_per_sm=active.value)
    return record


class CapturedLaunches:
    """Diagnostic graph owner; modules, buffers and parameter storage outlive it."""

    def __init__(self, lib, session, launch, nodes, keepalive):
        self.session = session
        self.keepalive = (launch, keepalive)
        self.graph, self.executable = ct.c_void_p(), ct.c_void_p()
        pointer = ct.c_void_p
        self.functions = {}
        signatures = {
            "cuStreamBeginCapture": [pointer, ct.c_int],
            "cuStreamEndCapture": [pointer, ct.POINTER(pointer)],
            "cuGraphInstantiateWithFlags": [ct.POINTER(pointer), pointer, ct.c_ulonglong],
            "cuGraphGetNodes": [pointer, pointer, ct.POINTER(ct.c_size_t)],
            "cuGraphLaunch": [pointer, pointer],
            "cuGraphExecDestroy": [pointer],
            "cuGraphDestroy": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = arguments, ct.c_int
            self.functions[name] = function
        self.call("cuStreamBeginCapture", pointer(session.stream), 1)
        try:
            for _ in range(nodes):
                launch()
        except BaseException:
            # End invalidated capture too, so the stream remains usable.
            self.functions["cuStreamEndCapture"](pointer(session.stream), ct.byref(self.graph))
            self.close()
            raise
        try:
            self.call("cuStreamEndCapture", pointer(session.stream), ct.byref(self.graph))
            count = ct.c_size_t()
            self.call("cuGraphGetNodes", self.graph, None, ct.byref(count))
            if count.value != nodes:
                raise RuntimeError(f"captured {count.value} nodes; expected {nodes}")
            self.nodes = count.value
            self.call("cuGraphInstantiateWithFlags", ct.byref(self.executable), self.graph, 0)
        except BaseException:
            self.close()
            raise

    def call(self, name, *arguments):
        result = self.functions[name](*arguments)
        if result:
            raise RuntimeError(f"{name} returned {result}")

    def launch(self):
        self.call("cuGraphLaunch", self.executable, ct.c_void_p(self.session.stream))

    def close(self):
        if self.session.synchronize():
            raise RuntimeError("graph cleanup synchronization failed")
        if self.executable.value:
            self.call("cuGraphExecDestroy", self.executable)
            self.executable = ct.c_void_p()
        if self.graph.value:
            self.call("cuGraphDestroy", self.graph)
            self.graph = ct.c_void_p()


def run(tool, storage, capacity, mode, samples, reps, reverse):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    bounded = compile_norm(tool, storage, capacity, mode, bounded=True)
    count = math.prod(capacity)
    lib, modules, graphs, records = driver(), [], [], []
    with NvidiaDeviceSession() as session:
        def load(compiled):
            ptx, metadata = compiled
            module, function = ct.c_void_p(), ct.c_void_p()
            encoded = ct.create_string_buffer(ptx.encode())
            started = time.perf_counter()
            if lib.cuModuleLoadData(ct.byref(module), encoded):
                raise RuntimeError("norm module load failed")
            modules.append(module)
            if lib.cuModuleGetFunction(ct.byref(function), module, metadata["entry"].encode()):
                raise RuntimeError("norm entry lookup failed")
            metadata = dict(metadata)
            metadata["module_load_host_ms"] = (time.perf_counter() - started) * 1e3
            metadata["driver_function_resources"] = function_resources(lib, function)
            return function, metadata

        try:
            bound_function, bound_receipt = load(bounded)
            for sq in ((capacity[2], 3, 1) if reverse else (1, 3, capacity[2])):
                shape = (*capacity[:2], sq, capacity[-1])
                rows, columns = math.prod(shape[:-1]), shape[-1]
                rng = np.random.default_rng(19102026 + sq)
                x = rng.normal(0, .5, shape).astype(dtype)
                xx = x.astype(np.float64)
                expected = (xx / np.sqrt(np.mean(xx * xx, axis=-1, keepdims=True)
                                         + float(np.float32(1e-5)))).reshape(-1)
                rounded = expected.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                             - rounded.astype(np.float64))
                budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                padded = np.zeros(count, dtype=dtype)
                padded[:x.size] = x.reshape(-1)
                a = session.upload(padded)
                static_function, static_receipt = load(compile_norm(
                    tool, storage, shape, mode, bounded=False))
                arms = {}
                grid = rows if mode == "cooperative_128" else (rows + 127) // 128
                for label, function, receipt in (("bounded", bound_function, bound_receipt),
                                                  ("static", static_function, static_receipt)):
                    out = session.upload(np.full(count + 16, 123, dtype=dtype))
                    values = [ct.c_uint64(a.ptr), ct.c_uint64(out.ptr), ct.c_int64(rows), ct.c_int64(columns)]
                    params = (ct.c_void_p * 4)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in values))

                    def launch(function=function, params=params):
                        if lib.cuLaunchKernel(function, grid, 1, 1, 128, 1, 1, 0,
                                              ct.c_void_p(session.stream), params, None):
                            raise RuntimeError("norm member launch failed")

                    launch()
                    if session.synchronize():
                        raise RuntimeError("norm synchronization failed")
                    record = {**receipt, "event_dispatch_samples_ms": [], "completed_raw_host_samples_ms": [],
                              "graph_dispatch_per_node_samples_ms": []}
                    arms[label] = (record, launch, out, values, params)

                def check():
                    for record, _, out, *_ in arms.values():
                        actual = session.download(out)
                        np.testing.assert_array_equal(actual[x.size:], dtype(123))
                        numeric = actual[:x.size].astype(np.float64)
                        if not np.isfinite(numeric).all() or not np.all(np.abs(numeric - expected) <= budget):
                            raise AssertionError("independent RMSNorm oracle mismatch")
                        record["max_abs_error"] = float(np.max(np.abs(numeric - expected)))
                    np.testing.assert_array_equal(session.download(arms["bounded"][2]).view(np.uint8),
                                                  session.download(arms["static"][2]).view(np.uint8))
                    np.testing.assert_array_equal(session.download(a).view(np.uint8), padded.view(np.uint8))

                check()
                captured = {}
                for label, arm in arms.items():
                    graph = CapturedLaunches(lib, session, arm[1], reps, arm)
                    graphs.append(graph)
                    captured[label] = graph
                    arm[0]["captured_kernel_nodes"] = graph.nodes
                for sample in range(samples):
                    order = ("bounded", "static") if (sample % 2 == 0) != reverse else ("static", "bounded")
                    for label in order:
                        record, launch, *_ = arms[label]
                        record["event_dispatch_samples_ms"].append(session.measure(launch, reps=reps, warmup=3))
                        record["graph_dispatch_per_node_samples_ms"].append(
                            session.measure(captured[label].launch, reps=8, warmup=3) / reps)
                        started = time.perf_counter()
                        for _ in range(reps):
                            launch()
                        if session.synchronize():
                            raise RuntimeError("completed norm window failed")
                        record["completed_raw_host_samples_ms"].append((time.perf_counter() - started) * 1e3 / reps)
                check()
                for record, *_ in arms.values():
                    record.update(graph_dispatch_per_node_median_ms=median(record["graph_dispatch_per_node_samples_ms"]),
                                  event_dispatch_median_ms=median(record["event_dispatch_samples_ms"]),
                                  completed_raw_host_median_ms=median(record["completed_raw_host_samples_ms"]),
                                  correctness="independent_oracle_and_raw_byte_parity_before_after")
                row = {"storage": storage, "mode": mode, "capacity_shape": list(capacity),
                       "runtime_shape": list(shape), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                       "arms": {label: value[0] for label, value in arms.items()}}
                records.append(row)
                print(storage, mode, shape,
                      {label: value[0]["event_dispatch_median_ms"] for label, value in arms.items()}, flush=True)
        finally:
            session.synchronize()
            for graph in reversed(graphs):
                graph.close()
            for module in reversed(modules):
                if module.value and lib.cuModuleUnload(module):
                    raise RuntimeError("norm module unload failed")
    assert len({r["arms"]["bounded"]["stage_sha256"]["ptx"] for r in records}) == 1
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error("at least three samples and positive reps required")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid",
         "--format=csv,noheader"], text=True).strip().splitlines()
    if len(devices) != 1 or devices[0].split(",")[1].strip() != "12.0":
        raise RuntimeError("requires one owning SM120 device")
    tool = find_tessera_opt()
    provider = rt._load_nvidia_gemm_runtime()
    if tool is None or provider is None:
        raise RuntimeError("requires matching compiler and native session runtime")
    paths = ("src/compiler/ir/include/Tessera/IR/BoundedQueryNormContract.h",
             "src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/programming_model/ir/ScheduleDialect.cpp",
             "src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td",
             "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
             "src/compiler/ir/TileOps.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
             "src/transforms/lib/TileToX86Pass.cpp",
             "tests/unit/test_sm120_bounded_norm_schedule.py",
             "tests/device/nvidia/test_bounded_norm_member.py",
             "benchmarks/nvidia/benchmark_bounded_query_norm.py")
    profiles = [(storage, capacity, mode) for storage in ("f16", "bf16")
                for capacity in ((2, 3, 7, 35), (1, 2, 17, 257), (1, 4, 257, 257))
                for mode in ("serial", "cooperative_128")]
    if args.reverse:
        profiles.reverse()
    packet = {"schema": "tessera.bounded_query_norm_member.v1", "device": devices[0],
              "route": "textual semantic Graph -> Schedule -> Tile -> NVIDIA LLVM/NVVM -> PTX",
              "scope": "raw member event dispatch, captured-graph dispatch per node, and completed raw host windows; not public/composed VJP or isolated profiler kernel time",
              "graph_replays_per_event_sample": 8,
              "graph_nodes": args.reps,
              "samples": args.samples, "reps": args.reps, "reverse_order": args.reverse,
              "compiler_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
              "provider_sha256": hashlib.sha256(Path(provider._name).read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
              "rows": []}
    for storage, capacity, mode in profiles:
        packet["rows"].extend(run(tool, storage, capacity, mode, args.samples, args.reps, args.reverse))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
