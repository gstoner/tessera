"""Matched ordinary-JIT manifest reconstruction cost on exact gfx1201."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_program import runtime_artifact
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle


class ManifestControl(dict):
    """Reconstruct the same compiler artifact, retaining identical native images."""
    rebuild = False

    def get(self, key, default=None):
        cached = super().get(key, default)
        if cached is not None and self.rebuild:
            program, _ = cached
            return program, runtime_artifact(program)
        return cached


def record(shape, reordered):
    arrays, _, _, _, expected = inputs_and_oracle(*shape)
    function = make_function(shape[1], shape[2], reordered)
    names = ("codes", "scales", "projection_globals", "a", "a_scale")
    inputs = dict(zip(names, arrays))
    function(**inputs)
    images = [p.image.image_digest for p in function.native_nvfp4_packages()]
    cache = ManifestControl(function._rocm_nvfp4_program_cache)
    function._rocm_nvfp4_program_cache = cache
    samples = {"retained_artifact": [], "rebuilt_manifest_control": []}
    for trial in range(7):
        order = list(samples)
        if trial % 2:
            order.reverse()
        for arm in order:
            cache.rebuild = arm == "rebuilt_manifest_control"
            start = time.perf_counter_ns()
            output = function(**inputs)
            samples[arm].append((time.perf_counter_ns() - start) / 1e6)
            np.testing.assert_allclose(output.astype(np.float32), expected,
                                       rtol=.008, atol=.015625)
            assert images == [p.image.image_digest for p in function.native_nvfp4_packages()]
    medians = {name: median(values) for name, values in samples.items()}
    return {"shape_mnk": list(shape), "reordered_arguments": reordered,
            "wall_samples_ms": samples, "wall_medians_ms": medians,
            "retained_over_rebuilt": medians["retained_artifact"] / medians["rebuilt_manifest_control"],
            "component_image_digests": images,
            "correctness": "independent folded oracle after every timed call",
            "timing_scope": "ordinary allocating JIT, identical Graph/images; control rebuilds manifest before the same checked runtime launch"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("exact gfx1201 required")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    name = C.create_string_buffer(256)
    device = C.c_int()
    if hip.hipGetDevice(C.byref(device)) or hip.hipDeviceGetName(name, len(name), device.value):
        raise RuntimeError("actual device identity required")
    root = Path(__file__).resolve().parents[2]
    sources = ("python/tessera/compiler/jit.py", "python/tessera/compiler/rocm_nvfp4_program.py",
               "python/tessera/runtime.py", "benchmarks/rocm/benchmark_nvfp4_manifest_cache.py")
    packet = {"architecture": "gfx1201", "device": name.value.decode(),
              "device_ordinal": device.value, "selector_promotion": False,
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
              "rows": [record(shape, reorder) for shape in
                       ((128, 32, 256), (257, 80, 1024), (256, 64, 64))
                       for reorder in (False, True)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{"shape": r["shape_mnk"], "reorder": r["reordered_arguments"],
                      "ratio": r["retained_over_rebuilt"]} for r in packet["rows"]], indent=2))


if __name__ == "__main__":
    main()
