"""Diagnostic native CPU materialization gate for mapped result permutations.

This does not select a production vmap route or establish GPU Schedule/Tile
execution. It times completed native JIT calls after typed Graph verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import numpy as np
import tessera as ts
from tessera import _jit_boundary as jit
from tessera.compiler.trace import trace, to_graph_ir_module


SOURCES = (
    "python/tessera/compiler/graph_ir.py", "python/tessera/compiler/trace.py",
    "src/compiler/ir/TesseraOps.cpp", "src/compiler/ir/TesseraOps.td",
    "src/compiler/ir/include/Tessera/IR/TransposeUtils.h",
    "src/compiler/ir/LinearTransposeInterface.cpp",
    "src/transforms/lib/TesseraToLinalgPass.cpp",
    "src/transforms/lib/SymbolicDimEqualityPass.cpp",
    "src/compiler/diagnostics/ShapeInferencePass.cpp",
    "python/tessera/compiler/diagnostic_codes.py",
    "tests/tessera-ir/phase2/sprint_v5_symdim_equality.mlir",
    "tests/unit/test_graph_transpose_axis_contract.py",
    "benchmarks/compiler/benchmark_native_transpose_axes.py",
)


def profile(shape, axes, opt):
    def source(x):
        return ts.ops.transpose(x, axes=axes)
    graph = to_graph_ir_module(trace(source, (shape, "fp32")), name="axes")
    semantic = graph.to_mlir(canonical=True)
    start = time.perf_counter()
    compiled = subprocess.run([opt, "--tessera-to-linalg"], input=semantic,
                              text=True, capture_output=True, check=True, timeout=90)
    handle = jit.compile_module(compiled.stdout)
    compile_ms = (time.perf_counter() - start) * 1000
    value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * .0625
    expected = value.transpose(axes).copy()
    output = np.empty(expected.shape, np.float32)
    try:
        before = jit.invocation_count()
        jit.invoke(handle, "axes", [value], [output])
        assert jit.invocation_count() == before + 1
        np.testing.assert_array_equal(output, expected)
        repetitions = 32
        while True:
            start = time.perf_counter()
            for _ in range(repetitions):
                jit.invoke(handle, "axes", [value], [output])
            if time.perf_counter() - start >= .050:
                break
            repetitions *= 2
            if repetitions > 65536:
                raise RuntimeError("could not establish a measurable completed-call window")
        samples = []
        count = jit.compile_count()
        for iteration in range(7):
            active = value * (1 if iteration % 2 == 0 else -.75)
            start = time.perf_counter()
            for _ in range(repetitions):
                jit.invoke(handle, "axes", [active], [output])
            samples.append((time.perf_counter() - start) * 1000 / repetitions)
            np.testing.assert_array_equal(output, active.transpose(axes))
        assert jit.compile_count() == count
    finally:
        jit.destroy(handle)
    return {"shape": shape, "axes": axes, "graph_ir": semantic,
            "linalg_ir": compiled.stdout, "compile_ms": compile_ms,
            "completed_native_call_samples_ms": samples,
            "completed_native_call_median_ms": median(samples),
            "calls_per_window": repetitions,
            "window_ms": [sample * repetitions for sample in samples],
            "correctness": "passed_before_and_after_timing",
            "scope": "synchronous CPU JIT ABI invocation and materialization; compilation excluded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opt = os.environ["TESSERA_OPT"]
    runtime = os.environ["TESSERA_JIT_LIB"]
    root = Path(__file__).resolve().parents[2]
    cpu = next(line.split(":", 1)[1].strip() for line in
               Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    packet = {"schema": "tessera.native_transpose_axes.cpu_diagnostic.v1",
              "cpu": cpu, "machine": os.uname().machine,
              "compiler_sha256": hashlib.sha256(Path(opt).read_bytes()).hexdigest(),
              "jit_runtime_sha256": hashlib.sha256(Path(runtime).read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                for name in SOURCES},
              "rows": [profile(shape, axes, opt) for shape, axes in (
                  ((2, 3, 5), (2, 0, 1)), ((3, 3, 3), (1, 2, 0)),
                  ((2, 3, 4, 5), (1, 3, 0, 2)), ((16, 64, 64), (2, 0, 1)),
                  ((8, 16, 32, 32), (1, 3, 0, 2)))],
              "closure": "diagnostic Graph/Linalg/LLVM materialization gate; GPU Schedule/Tile mapped output route remains open"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    for row in packet["rows"]:
        print(row["shape"], row["axes"], row["completed_native_call_median_ms"])


if __name__ == "__main__":
    main()
