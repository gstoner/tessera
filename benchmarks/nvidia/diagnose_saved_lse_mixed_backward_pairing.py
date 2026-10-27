"""Diagnostic: paired, interleaved device-event medians for mixed Q vs K/V reverse.

SM120-SAVED-LSE-MIXED-BACKWARD-20261026. Every arm, homogeneous controls
included, is compiled by the public JIT (``autodiff="reverse"``), checked
against the fp64 reverse oracle, and then executed through the same prepared
reverse owner the public call uses. The owner's CUDA-event windows are
recorded per call: the saved-LSE forward, the row-correction producer (row
programs only) and the backward kernel. Arms
alternate ABAB/BABA. CUDA events alone are diagnostic on sm_120, never a
promotion witness.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess

import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler import prepared_attention_vjp
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--reps", type=int, default=5)
# The typed (half) reverse kernel takes seconds per call at the large shape
# (pre-existing on main; see the packet README), so that shape is sampled
# with fewer alternating rounds of one call each.
parser.add_argument("--large-rounds", type=int, default=6)
args = parser.parse_args()


def attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def dt(name):
    return np.dtype(ml_dtypes.bfloat16 if name == "bfloat16" else name)


#: (label, Q storage, K/V storage, row-correction policy). "auto" is the
#: default selection; the two "inline" arms force the per-element correction
#: as before/after controls inside the same interleaved rounds.
ARMS = [
    ("f16 homogeneous", "float16", "float16", "auto"),
    ("f16 / bf16", "float16", "bfloat16", "auto"),
    ("bf16 homogeneous", "bfloat16", "bfloat16", "auto"),
    ("bf16 / f16", "bfloat16", "float16", "auto"),
    ("f32 homogeneous", "float32", "float32", "auto"),
    ("f32 / f16", "float32", "float16", "auto"),
    ("f32 / bf16", "float32", "bfloat16", "auto"),
    ("f16 / f32", "float16", "float32", "auto"),
    ("f16 homogeneous, inline", "float16", "float16", "inline"),
    ("f32 / f16, inline", "float32", "float16", "inline"),
]


def owner_for(metadata, inputs):
    owner = prepared_attention_vjp.prepared(metadata)
    dims = owner.runtime_dimensions(inputs)
    return owner if dims == owner.dims else prepared_attention_vjp.prepared(metadata, runtime_dims=dims)


out = {}
SHAPES = (((1, 2, 1, 3, 5, 4, 3), args.rounds, args.reps),
          ((1, 8, 2, 128, 512, 64, 64), args.large_rounds, 1))
for shape, rounds, reps in SHAPES:
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(1)
    base = {"q": rng.normal(0, .3, (b, hq, sq, d)), "k": rng.normal(0, .3, (b, hkv, sk, d)),
            "v": rng.normal(0, .3, (b, hkv, sk, dv))}
    do_base = rng.normal(0, .2, (b, hq, sq, dv))
    arms, functions = {}, []
    for label, q_name, kv_name, policy in ARMS:
        os.environ["TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION"] = policy
        values = {"q": base["q"].astype(dt(q_name)), "k": base["k"].astype(dt(kv_name)),
                  "v": base["v"].astype(dt(kv_name))}
        do = do_base.astype(dt(q_name))
        fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("q", "k", "v"))(attention)
        functions.append(fn)
        gradients = fn.native_backward(**values, out_cotangents=do)
        assert fn.last_backward_execution["execution_kind"] == "native_gpu"
        _, expected = reference(values["q"], values["k"], values["v"], do, None,
                                np.zeros((b, hq, sq), np.float32), causal=True)
        for index, gradient in enumerate(gradients):
            compare(gradient, expected[index])
        metadata = fn.native_backward_runtime_artifact().metadata
        inputs = tuple(np.ascontiguousarray(x) for x in (*(values[n] for n in fn.arg_names), do))
        arms[label] = (metadata, inputs)
    samples = {label: {"forward": [], "row_producer": [], "backward": [], "reverse_total": []} for label in arms}
    for round_ in range(rounds):
        order = list(arms) if round_ % 2 == 0 else list(arms)[::-1]
        for label in order:
            metadata, inputs = arms[label]
            owner = owner_for(metadata, inputs)
            for _ in range(reps):
                owner.invoke(metadata, inputs)
                times = owner.last_device_ms
                # Row programs (SM120-TYPED-REVERSE-ROW-PREPASS-20261026) report
                # forward, row producer, backward and the whole device window;
                # the inline reverse reports forward and backward.
                row = times[1] if owner.row_program else 0.0
                backward = times[2] if owner.row_program else times[1]
                samples[label]["forward"].append(times[0] * 1000)
                samples[label]["row_producer"].append(row * 1000)
                samples[label]["backward"].append(backward * 1000)
                samples[label]["reverse_total"].append((row + backward) * 1000)
    out[str(shape)] = {label: {window: round(statistics.median(s), 2) for window, s in windows.items()}
                       for label, windows in samples.items()}
    for fn in functions:
        fn.close_native_storage()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,compute_cap",
                               "--format=csv,noheader"], text=True).strip()
args.output.write_text(json.dumps({"commit": commit, "worktree_dirty": bool(status), "device": gpu,
    "timing": ("prepared reverse owner CUDA events: saved-LSE forward, row-correction producer (row programs; 0 inline), "
               "backward kernel, and reverse_total = producer + backward; "
               f"small shape {args.reps} calls x {args.rounds} alternating rounds, large shape "
               f"1 call x {args.large_rounds} alternating rounds; median us"),
    "correctness": "fp64 reverse oracle on dQ/dK/dV before timing",
    "medians_us": out}, indent=1) + "\n")
print(json.dumps(out, indent=1))
