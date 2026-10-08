"""Measure version metadata reuse on existing compiler-owned gfx1201 RMSNorm."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native as native, scheduled_kernel
from tessera.compiler.native_unary_contract import verify_unary_ancestry
from tests.unit.test_rocm_gfx1201_scheduled import _rmsnorm_graph


def check(package, shape, dtype, seed):
    x = np.random.default_rng(seed).normal(size=shape).astype(dtype)
    output = np.zeros_like(x)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir)
    result = rt.launch(artifact, {"buffers": {"x": x, "o": output},
        "scalars": {"Rows": shape[0], "K": shape[1], "Epsilon": 1.0e-5}})
    if not result.get("ok") or result.get("execution_kind") != "native_gpu":
        raise RuntimeError(str(result))
    oracle_x = x.astype(np.float64)
    expected = oracle_x / np.sqrt(np.mean(oracle_x**2, axis=-1, keepdims=True) + 1e-5)
    tol = 2e-3 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(output, expected, rtol=tol, atol=tol)
    return float(np.max(np.abs(output.astype(np.float64) - expected)))


def record():
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("exact gfx1201 device required")
    rows = []
    for cache_mode in ("empty_image_caches", "reused_image_caches"):
        for shape in ((5, 32), (17, 128), (200, 1024)):
            for dtype_name, dtype in (("fp16", np.float16), ("fp32", np.float32)):
                artifact = scheduled_kernel.lower_scheduled_kernel(
                    _rmsnorm_graph(dtype=dtype_name, shape=shape), target="rocm_gfx1201")
                verify_unary_ancestry(artifact, target="rocm", architecture="gfx1201")
                def package():
                    return native.package_scheduled_kernel(
                        artifact, pipeline_name="tessera-lower-to-rocm")
                prepared = package()
                errors = [check(prepared, shape, dtype, 1201)]
                samples = {"uncached_versions": [], "cached_versions": []}
                counts = {arm: [] for arm in samples}
                identities = {}
                for trial in range(7):
                    arms = tuple(samples) if trial % 2 == 0 else tuple(reversed(samples))
                    for arm in arms:
                        retained = native._VERSION_FINGERPRINTS
                        original = subprocess.run
                        calls = [0]
                        if arm == "uncached_versions":
                            native._VERSION_FINGERPRINTS = {}
                        def observed(*args, **kwargs):
                            calls[0] += 1
                            return original(*args, **kwargs)
                        subprocess.run = observed
                        try:
                            if cache_mode == "empty_image_caches":
                                native._cache.clear()
                            start = time.perf_counter()
                            candidate = package()
                            samples[arm].append((time.perf_counter() - start) * 1e3)
                        finally:
                            subprocess.run = original
                            native._VERSION_FINGERPRINTS = retained
                        counts[arm].append(calls[0])
                        identities[arm] = (candidate.image.compiler_fingerprint,
                                          candidate.image.toolchain_fingerprint,
                                          hashlib.sha256(candidate.image.payload).hexdigest())
                        errors.append(check(candidate, shape, dtype, 1202 + trial))
                assert identities["uncached_versions"] == identities["cached_versions"]
                medians = {arm: median(v) for arm, v in samples.items()}
                rows.append({"image_cache_mode": cache_mode, "shape": list(shape), "dtype": dtype_name,
                    "samples_ms": samples, "subprocess_counts": counts,
                    "medians_ms": medians,
                    "uncached_over_cached": medians["uncached_versions"] / medians["cached_versions"],
                    "max_abs_error": max(errors), "fingerprints_and_images_match": True})
                print("passed", len(rows), "/12", flush=True)
    return {"architecture": rt._rocm_live_arch(), "rows": rows,
        "measurement": "native package wall time; version metadata varied within explicit image-cache modes",
        "source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(native._tessera_opt().read_bytes()).hexdigest(),
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["not kernel timing", "existing RMSNorm profile only",
            "compiler binary comes from five-slice integration build, adapter from isolated PR"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(record(), indent=2) + "\n")
