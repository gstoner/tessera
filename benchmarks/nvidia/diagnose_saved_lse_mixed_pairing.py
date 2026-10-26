"""Diagnostic: paired, interleaved CUDA-event medians, mixed vs homogeneous saved-LSE.

SM120-SAVED-LSE-MIXED-STORAGE-20261025. The packet recorder times its rows one
after another, so device clock drift lands between storage arms. This script
alternates every arm (ABAB/BABA) inside one loop after an fp64 check. CUDA
events alone are diagnostic on sm_120, never a promotion witness.
"""
import argparse, json, pathlib, statistics, subprocess
import ml_dtypes, numpy as np, tessera as ts
from tests.device.nvidia.test_saved_lse_storage import saved, reference, compare

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--fp8", action="store_true",
                    help="time unscaled FP8 K/V arms (SM120-SAVED-LSE-FP8-KV-20261025) beside their controls")
args = parser.parse_args()


def dt(n):
    if n in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        return np.dtype(getattr(ml_dtypes, n))
    return np.dtype(n)
ARMS = [("float16", "float16"), ("float16", "bfloat16"), ("bfloat16", "bfloat16"), ("bfloat16", "float16"),
        ("float32", "float32"), ("float32", "float16"), ("float32", "bfloat16"), ("float16", "float32")]
if args.fp8:
    ARMS = [("float16", "float16"), ("float16", "float8_e4m3fn"), ("float16", "float8_e5m2"),
            ("bfloat16", "bfloat16"), ("bfloat16", "float8_e4m3fn"), ("bfloat16", "float8_e5m2"),
            ("float32", "float32"), ("float32", "float8_e4m3fn"), ("float32", "float8_e5m2")]
out = {}
for shape in ((1, 2, 1, 3, 5, 4, 3), (1, 8, 2, 128, 512, 64, 64)):
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(1)
    base = {"q": rng.normal(0, .2, (b, hq, sq, d)), "k": rng.normal(0, .2, (b, hkv, sk, d)),
            "v": rng.normal(0, .2, (b, hkv, sk, dv))}
    fns, vals = {}, {}
    for q, kv in ARMS:
        values = {n: x.astype(dt(q) if n == "q" else dt(kv)) for n, x in base.items()}
        fn = ts.jit(target="nvidia_sm120")(saved)
        pair = fn(**values)
        compare(pair, reference(values, True), dt(q))
        assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
        fns[(q, kv)], vals[(q, kv)] = fn, values
    samples = {arm: [] for arm in ARMS}
    for rep in range(40):
        order = ARMS if rep % 2 == 0 else ARMS[::-1]
        for arm in order:
            fns[arm](**vals[arm])
            owner = next(iter(fns[arm]._native_prepared_attention_calls.values()))
            samples[arm].append(owner.last_device_ms * 1000)
    out[str(shape)] = {f"{q}/{kv}": round(statistics.median(s), 2) for (q, kv), s in samples.items()}
    for fn in fns.values(): fn.close_native_storage()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,compute_cap",
                               "--format=csv,noheader"], text=True).strip()
args.output.write_text(json.dumps({"commit": commit, "worktree_dirty": bool(status), "device": gpu,
    "timing": "CUDA events around the retained kernel launch; 40 alternating rounds; median us",
    "correctness": "fp64 oracle before timing", "medians_us": out}, indent=1) + "\n")
print(json.dumps(out, indent=1))
