"""Diagnostic: paired, interleaved kernel-event medians for scaled FP8 K/V saved-LSE.

SM120-SAVED-LSE-SCALED-FP8-KV-20261025. Every arm, homogeneous controls
included, is compiled by the public JIT, checked against an fp64 oracle, and
then timed through the same checked launcher benchmark entry
(``tessera_nvidia_ptx_benchmark``: CUDA events around ``reps`` launches after
``warmup``). Arms alternate ABAB/BABA. CUDA events alone are diagnostic on
sm_120, never a promotion witness.
"""
import argparse
import ctypes
import json
import pathlib
import statistics
import subprocess

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_saved_lse_storage import compare, reference, saved

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--reps", type=int, default=10)
args = parser.parse_args()


def scaled(q, k8, v8, ks, vs):
    k = ts.ops.dequantize_scaled(k8, ks)
    v = ts.ops.dequantize_scaled(v8, vs)
    return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")


def dt(name):
    return np.dtype(getattr(ml_dtypes, name) if name in ("bfloat16", "float8_e4m3fn", "float8_e5m2") else name)


#: (label, Q storage, K/V storage, scale granularity or None).
ARMS = [
    ("f16 homogeneous", "float16", "float16", None),
    ("f16 / e4m3 unscaled", "float16", "float8_e4m3fn", None),
    ("f16 / e4m3 per-tensor", "float16", "float8_e4m3fn", "per_tensor"),
    ("f16 / e4m3 per-token", "float16", "float8_e4m3fn", "per_token"),
    ("f16 / e4m3 per-element", "float16", "float8_e4m3fn", "full"),
    ("f16 / e5m2 unscaled", "float16", "float8_e5m2", None),
    ("f16 / e5m2 per-token", "float16", "float8_e5m2", "per_token"),
]


def scale_shapes(granularity, hkv, sk, d, dv):
    return {"per_tensor": ((1, 1, 1, 1), (1, 1, 1, 1)),
            "per_token": ((1, hkv, sk, 1), (1, hkv, sk, 1)),
            "full": ((1, hkv, sk, d), (1, hkv, sk, dv))}[granularity]


lib = rt._load_nvidia_ptx_launch()
if lib is None:
    raise RuntimeError("libtessera_nvidia_ptx_launch.so not loadable")
out = {}
for shape in ((1, 2, 1, 3, 5, 4, 3), (1, 8, 2, 128, 512, 64, 64)):
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(1)
    base = {"q": rng.normal(0, .2, (b, hq, sq, d)), "k": rng.normal(0, 2, (b, hkv, sk, d)),
            "v": rng.normal(0, 2, (b, hkv, sk, dv))}
    launches = {}
    for label, q_name, kv_name, granularity in ARMS:
        if granularity is None:
            values = {"q": base["q"].astype(dt(q_name)), "k": base["k"].astype(dt(kv_name)),
                      "v": base["v"].astype(dt(kv_name))}
            fn = ts.jit(target="nvidia_sm120")(saved)
            pair = fn(**values)
            compare(pair, reference(values, True), dt(q_name))
        else:
            ks_shape, vs_shape = scale_shapes(granularity, hkv, sk, d, dv)
            values = {"q": base["q"].astype(dt(q_name)), "k8": base["k"].astype(dt(kv_name)),
                      "v8": base["v"].astype(dt(kv_name)),
                      "ks": rng.uniform(.05, .3, ks_shape).astype(np.float32),
                      "vs": rng.uniform(.05, .3, vs_shape).astype(np.float32)}
            fn = ts.jit(target="nvidia_sm120")(scaled)
            pair = fn(**values)
            dense = {"q": values["q"], "k": ts.ops.dequantize_scaled(values["k8"], values["ks"]),
                     "v": ts.ops.dequantize_scaled(values["v8"], values["vs"])}
            compare(pair, reference(dense, True), dt(q_name))
        assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
        artifact = fn._cached_artifact
        descriptor, image = artifact.launch_descriptor, artifact.native_image
        entry = descriptor.entry_symbol
        if entry not in rt._nvidia_ptx_registered and rt._register_nvidia_ptx(
                lib, entry, image.payload.decode("ascii")) != 0:
            raise RuntimeError(f"ptx register failed for {entry}")
        bindings = sorted(descriptor.buffers, key=lambda binding: binding.ordinal)
        inputs = [binding for binding in bindings if binding.direction == "input"]
        arrays = [np.ascontiguousarray(values[fn.arg_names[int(binding.name[1:])]]) for binding in inputs]
        arrays += [np.empty(o.shape, o.dtype) for o in pair]
        scalar_names = [s.name for s in sorted(descriptor.scalars, key=lambda value: value.ordinal)]
        provenance = descriptor.provenance
        dims = list(provenance["shape"]) + (list(provenance["bias_shape"]) if "BiasB" in scalar_names else [])
        dims += list(provenance.get("kv_scale_shapes", []))
        launches[label] = (entry, arrays, dims, fn)
    samples = {label: [] for label in launches}
    for round_ in range(args.rounds):
        order = list(launches) if round_ % 2 == 0 else list(launches)[::-1]
        for label in order:
            entry, arrays, dims, _ = launches[label]
            buffers = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
            scalars = (ctypes.c_int64 * len(dims))(*dims)
            latency = ctypes.c_float()
            rc = lib.tessera_nvidia_ptx_benchmark(entry.encode(), buffers, len(arrays), scalars,
                                                  len(dims), 2, args.reps, ctypes.byref(latency))
            if rc:
                raise RuntimeError(f"benchmark {label} rc={rc}")
            samples[label].append(latency.value * 1000)
    out[str(shape)] = {label: round(statistics.median(s), 2) for label, s in samples.items()}
    for _, _, _, fn in launches.values():
        fn.close_native_storage()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,compute_cap",
                               "--format=csv,noheader"], text=True).strip()
args.output.write_text(json.dumps({"commit": commit, "worktree_dirty": bool(status), "device": gpu,
    "timing": (f"tessera_nvidia_ptx_benchmark CUDA events: 2 warmup + {args.reps} launches per sample; "
               f"{args.rounds} alternating rounds; median us per launch"),
    "correctness": "fp64 oracle before timing (scaled arms: dequantize_scaled reference K/V)",
    "medians_us": out}, indent=1) + "\n")
print(json.dumps(out, indent=1))
