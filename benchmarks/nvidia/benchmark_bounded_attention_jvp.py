"""Exact SM120 bounded saved-LSE JVP: one image, multiple actual sequence sizes."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from benchmarks.record_device_ring_protocol import Device
from benchmarks.record_jit_attention_program import timed_native_product
from tessera.compiler.native_attention_program import compile_attention_program, NativeAttentionJVPProgram
from tessera.compiler.native_device_tape import _Buffer


def source(active, causal, bias=None):
    qt, kt, vt, ot = ("tensor<1x2x?x4xf32>", "tensor<1x1x?x4xf32>",
                      "tensor<1x1x?x3xf32>", "tensor<1x2x?x3xf32>")
    names, types = ["q", "k", "v"], [qt, kt, vt]
    if bias:
        names.append("bias")
        types.append("tensor<1x2x?x?xf32>" if bias == "full" else "tensor<1x2x1x1xf32>")
    arguments = ", ".join("%"+name+": "+typ for name, typ in zip(names, types, strict=True))
    operands = ", ".join("%"+name for name in names)
    return f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120",
      tessera.attention_shape_bounds = array<i64: 1, 2, 1, 9, 11, 4, 3>}} {{
      func.func @attention({arguments}) -> {ot}
        attributes {{tessera.autodiff = "forward",
                     tessera.autodiff.wrt_indices = [{", ".join(map(str, active))}]}} {{
        %o = "tessera.flash_attn"({operands}) {{causal = {str(causal).lower()},
          head_dim = 4 : i64, operandSegmentSizes = array<i32: 1, 1, 1, {int(bool(bias))}>}}
          : ({", ".join(types)}) -> {ot}
        return %o : {ot}
      }}
    }}"""


def oracle(values, directions, causal):
    q, k, v = (x.astype(np.float64) for x in values[:3])
    dq, dk, dv = (x.astype(np.float64) for x in directions[:3])
    k, v, dk, dv = (np.repeat(x, 2, axis=1) for x in (k, v, dk, dv))
    score = .5 * (q @ k.swapaxes(-1, -2))
    tangent = .5 * (dq @ k.swapaxes(-1, -2) + q @ dk.swapaxes(-1, -2))
    if len(values) == 4:
        score += values[3]
        tangent += directions[3]
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        legal = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk - sq, 0)
        score = np.where(legal, score, -np.inf)
    maximum = score.max(axis=-1, keepdims=True)
    weights = np.exp(score - maximum)
    total = weights.sum(axis=-1, keepdims=True)
    p = weights / total
    dp = p * (tangent - (p * tangent).sum(axis=-1, keepdims=True))
    return p @ v, (maximum + np.log(total))[..., 0], dp @ v + p @ dv


def execute(device, program, sq, sk, causal, bias):
    rng = np.random.default_rng(901 + sq * 31 + sk)
    shapes = [(1, 2, sq, 4), (1, 1, sk, 4), (1, 1, sk, 3)]
    if bias:
        shapes.append((1, 2, sq, sk) if bias == "full" else (1, 2, 1, 1))
    values = [rng.normal(size=s).astype(np.float32) * .2 for s in shapes]
    directions = [rng.normal(size=s).astype(np.float32) * .1 if i in program.active
                  else np.zeros(s, np.float32) for i, s in enumerate(shapes)]
    primal, lse, expected = oracle(values, directions, causal)
    step = 1e-4
    plus = [x.astype(np.float64) + step*d for x, d in zip(values, directions, strict=True)]
    minus = [x.astype(np.float64) - step*d for x, d in zip(values, directions, strict=True)]
    fd = (oracle(plus, directions, causal)[0] - oracle(minus, directions, causal)[0]) / (2 * step)
    np.testing.assert_allclose(expected, fd, rtol=2e-7, atol=2e-9)
    pointers = []

    def upload(x):
        ptr = ct.c_void_p()
        device.check(device.alloc(ct.byref(ptr), x.nbytes))
        pointers.append(ptr)
        device.check(device.htod(ptr, x.ctypes.data, x.nbytes))
        return SimpleNamespace(__cuda_array_interface__={
            "version": 3, "shape": x.shape, "typestr": x.dtype.str, "data": (ptr.value, False)})

    def download(x):
        spec = x.__cuda_array_interface__
        host = np.empty(spec["shape"], np.float32)
        device.check(device.dtoh(host.ctypes.data, ct.c_void_p(spec["data"][0]), host.nbytes))
        return host

    try:
        inputs = [upload(x) for x in values]
        tangents = [upload(directions[i]) for i in program.active]
        with program.capture(*inputs) as frame:
            np.testing.assert_allclose(download(frame.primal), primal, atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(download(frame._frame._saved[4]), lse, atol=3e-5, rtol=3e-5)
            result = frame.jvp(*tangents)
            actual = download(result)
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            doubled = frame.jvp(*(upload(2*directions[i]) for i in program.active))
            np.testing.assert_allclose(download(doubled), 2*expected, atol=3e-5, rtol=3e-5)
            np.testing.assert_array_equal(download(result), actual)
            # Time into separate frame-owned output storage, retaining the
            # original numerical result across subsequent native launches.
            direction_buffers = dict(frame._zeros)
            direction_buffers.update(zip(program.active, tangents, strict=True))
            tape = frame._frame
            out = _Buffer(tape, tape.shapes[3])
            extra = (tape._bias, direction_buffers[3]) if bias else ()
            raw, _, grid, _, _ = tape._jvp_binding.prepare(
                *tape._saved, *(direction_buffers[i] for i in range(3)),
                *extra, out, 128, sq, sk)
            assert grid == (2*sq, 1, 1)
            kernel = timed_native_product(device, program.tangent, raw, grid[0])
            np.testing.assert_array_equal(download(result), actual)
        wall = []
        for _ in range(3):
            start = time.perf_counter()
            with program.capture(*inputs) as fresh:
                fresh.jvp(*tangents)
            wall.append((time.perf_counter() - start)*1e3)
        return dict(actual_sq_sk=[sq, sk], correctness="passed_before_timing",
            max_abs_error=float(np.max(np.abs(actual-expected))),
            forward_lse_checked=True, retained_result_checked=True,
            kernel=kernel, capture_jvp_close_wall_samples_ms=wall)
    finally:
        for ptr in pointers:
            device.check(device.free(ptr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    device, rows = Device("nvidia"), []
    profiles = [((0,), True, None)] if args.smoke else [
        (active, causal, None) for active in ((0,), (1,), (2,), (2, 0, 1))
        for causal in (False, True)] + [
        ((3,), True, "broadcast"), ((0, 1, 2, 3), False, "full")]
    for active, causal, bias in profiles:
        text = source(active, causal, bias)
        program = compile_attention_program(text, active,
            compiler=Path(os.environ["TESSERA_OPT"]), llvm_bin=Path("/usr/lib/llvm-23/bin"),
            input_names=("q", "k", "v") + (("bias",) if bias else ()))
        encoded = program.to_json()
        digest = program.program_digest
        program = NativeAttentionJVPProgram.from_json(encoded, expected_digest=digest)
        assert program.to_json() == encoded
        stem = "_".join(map(str, active)) + "_" + str(causal) + "_" + str(bias)
        (artifacts/(stem+".mlir")).write_text(text)
        (artifacts/(stem+".program.json")).write_text(encoded)
        cases = [execute(device, program, sq, sk, causal, bias)
                 for sq, sk in ((1, 1), (3, 7), (7, 3), (9, 11))]
        assert program.program_digest == digest
        rows.append(dict(active=active, causal=causal, bias=bias,
            program_digest=digest, forward_image=program.pair.forward.image.image_digest,
            tangent_image_sha256=hashlib.sha256(program.tangent.image).hexdigest(),
            image_reuse_across_shapes=True, cases=cases))
        print("verified", stem, flush=True)
    paths = ("src/compiler/programming_model/lib/NativeAttentionJvp.h",
             "python/tessera/compiler/native_attention_program.py",
             "python/tessera/compiler/native_gpu_tensor.py",
             "benchmarks/nvidia/benchmark_bounded_attention_jvp.py")
    receipt = dict(device=gpu, source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=True, compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        route="textual Graph -> native AD -> Schedule -> Tile -> NVIDIA Target/LLVM/PTX -> checked resident ABI",
        timing_scope="preloaded JVP CUDA events separate from capture/JVP/close host wall",
        rows=rows, performance_promotion=False)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
