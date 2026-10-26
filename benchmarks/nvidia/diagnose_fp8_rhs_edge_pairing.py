"""Diagnostic: paired, interleaved device-event medians for the FP8 RHS edge.

SM120-W11-FP8-RHS-20261026. Each arm is the resident two-package edge
``matmul(lhs, f(rmsnorm(source)))``: the half edge stores the normalized
source in f16, the FP8 arms store quantize_scaled codes and decode them in the
matmul's B gather. Every arm is correctness-gated before timing, then timed
with session CUDA events around repeated stream launches (producer, consumer,
and both) in ABAB/BABA order. CUDA events alone are diagnostic on sm_120,
never a promotion witness; they include the host enqueue gap between launches.
"""
import argparse
import json
import pathlib
import statistics
import subprocess

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tessera.compiler.nvidia_tensor_rhs import package_norm_rhs_matmul

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--reps", type=int, default=20)
args = parser.parse_args()
PIPELINE = "tessera-nvidia-pipeline-sm120"


def edge_modules(fmt, m, k, n):
    x = tensor_ir_type((k, n), "fp16")
    a = tensor_ir_type((m, k), "fp16")
    d = tensor_ir_type((m, n), "fp32")
    norm = IROp(result="y", op_name="tessera.rmsnorm", operands=["%x"], operand_types=[str(x)],
                result_type=str(x), kwargs={"eps": 1e-5})
    if fmt is None:
        producer = GraphIRFunction(name="rhs", args=[IRArg("x", x)], result_types=[x],
                                   body=[norm], return_values=["%y"])
        matmul = IROp(result="d", op_name="tessera.matmul", operands=["%a", "%y"],
                      operand_types=[str(a), str(x)], result_type=str(d), kwargs={"rhs_storage_order": "row_major"})
        consumer = GraphIRFunction(name="mm", args=[IRArg("a", a), IRArg("y", x)], result_types=[d],
                                   body=[matmul], return_values=["%d"])
    else:
        s = tensor_ir_type((1, 1), "fp32")
        q = tensor_ir_type((k, n), fmt)
        b = tensor_ir_type((k, n), "fp32")
        quantize = IROp(result="q", op_name="tessera.quantize_scaled", operands=["%y", "%s"],
                        operand_types=[str(x), str(s)], result_type=str(q), kwargs={"format": fmt})
        producer = GraphIRFunction(name="rhs", args=[IRArg("x", x), IRArg("s", s)], result_types=[q],
                                   body=[norm, quantize], return_values=["%q"])
        dequant = IROp(result="b", op_name="tessera.dequantize_scaled", operands=["%q", "%s"],
                       operand_types=[str(q), str(s)], result_type=str(b), kwargs={})
        matmul = IROp(result="d", op_name="tessera.matmul", operands=["%a", "%b"],
                      operand_types=[str(a), str(b)], result_type=str(d), kwargs={"rhs_storage_order": "row_major"})
        consumer = GraphIRFunction(name="mm", args=[IRArg("a", a), IRArg("q", q), IRArg("s", s)],
                                   result_types=[d], body=[dequant, matmul], return_values=["%d"])
    return GraphIRModule(functions=[producer]), GraphIRModule(functions=[consumer])


def artifact(package):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
                              launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                              target_ir=package.target_ir)


out = {}
for shape in ((16, 4096, 4096), (256, 4096, 4096)):
    m, k, n = shape
    rng = np.random.default_rng(7)
    a_host = (rng.normal(size=(m, k)) * .05).astype(np.float16)
    x_host = rng.normal(size=(k, n)).astype(np.float16)
    s_host = np.full((1, 1), 1 / 32, np.float32)
    session = NvidiaDeviceSession()
    arms = {}
    try:
        for label, fmt in (("f16 edge", None), ("fp8_e4m3 edge", "fp8_e4m3"), ("fp8_e5m2 edge", "fp8_e5m2")):
            program = package_norm_rhs_matmul(*edge_modules(fmt, m, k, n), pipeline_name=PIPELINE)
            edge_dtype = (np.dtype(np.float16) if fmt is None else
                          np.dtype(ml_dtypes.float8_e4m3fn if fmt == "fp8_e4m3" else ml_dtypes.float8_e5m2))
            x = session.upload(x_host)
            a = session.upload(a_host)
            s = session.upload(s_host) if fmt else None
            edge = session.empty((k, n), edge_dtype)
            result = session.empty((m, n), np.float32)
            pa, ca = program.arguments(x, a, edge, result, s)
            producer, consumer = artifact(program.producer), artifact(program.consumer)

            def run(package, arguments):
                receipt = rt.launch(package, arguments, stream=session.stream)
                if receipt.get("ok") is not True:
                    raise RuntimeError(receipt)
            run(producer, pa)
            run(consumer, ca)
            assert session.synchronize() == 0
            got = session.download(result)
            codes = session.download(edge)
            decoded = codes.astype(np.float64) if fmt is None else ts.ops.dequantize_scaled(codes, s_host).astype(np.float64)
            expected = a_host.astype(np.float64) @ decoded
            bound = 1e-5 * (np.abs(a_host.astype(np.float64)) @ np.abs(decoded)) + 1e-30
            assert np.all(np.abs(got - expected) <= bound), label
            arms[label] = (producer, pa, consumer, ca, run)
        samples = {label: {"producer": [], "consumer": [], "edge": []} for label in arms}
        for round_ in range(args.rounds):
            order = list(arms) if round_ % 2 == 0 else list(arms)[::-1]
            for label in order:
                producer, pa, consumer, ca, run = arms[label]
                samples[label]["producer"].append(1000 * session.measure(lambda: run(producer, pa), reps=args.reps, warmup=2))
                samples[label]["consumer"].append(1000 * session.measure(lambda: run(consumer, ca), reps=args.reps, warmup=2))
                samples[label]["edge"].append(1000 * session.measure(
                    lambda: (run(producer, pa), run(consumer, ca)), reps=args.reps, warmup=2))
        out[str(shape)] = {label: {window: round(statistics.median(v), 2) for window, v in windows.items()}
                           for label, windows in samples.items()}
    finally:
        session.close()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,compute_cap",
                               "--format=csv,noheader"], text=True).strip()
args.output.write_text(json.dumps({"commit": commit, "worktree_dirty": bool(status), "device": gpu,
    "timing": (f"session CUDA events around {args.reps} resident stream launches after 2 warmup; "
               f"{args.rounds} alternating rounds; median us per launch (includes host enqueue gaps)"),
    "correctness": "fp64 bound 1e-5 * |A| @ |decode(edge)| on every arm before timing",
    "shape_order": "(M, K, N): lhs MxK f16, normalized source KxN, output MxN f32",
    "medians_us": out}, indent=1) + "\n")
print(json.dumps(out, indent=1))
