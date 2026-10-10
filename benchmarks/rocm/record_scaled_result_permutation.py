"""Correctness-gated gfx1201 scaled-product/result-permutation attribution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import numpy as np

from tessera import runtime
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram, package_native_scaled_primal)
from tests.unit.test_native_scaled_result_permutation import source
from tests.unit.test_rocm_independent_scaled_batch import batch_inputs


def record(shape, library):
    arrays, oracle = batch_inputs(shape=(1, *shape), fmt="e8m0",
                                 nk=False, policy="independent_rhs")
    arrays = tuple(value[0] for value in arrays)
    expected = np.ascontiguousarray(oracle[0].T)
    start = time.perf_counter()
    package = package_native_scaled_primal(source(shape))
    compile_ms = (time.perf_counter() - start) * 1e3
    with PreparedScaledProgram(package, arrays, runtime_library=library) as owner:
        generation, _ = owner.invoke()
        result = owner.read(generation)[0]
        np.testing.assert_allclose(result, expected, rtol=4e-5, atol=2e-5)
        error = float(np.max(np.abs(result - expected)))
        repeats = 1024
        while True:
            generation, values = owner.profile_members(repeats=repeats)
            np.testing.assert_allclose(owner.read(generation)[0], expected,
                                       rtol=4e-5, atol=2e-5)
            if min(values) * repeats >= 20 or repeats == 65536:
                break
            repeats *= 2
        captured = []
        for _ in range(5):
            generation, values = owner.profile_members(repeats=repeats)
            np.testing.assert_allclose(owner.read(generation)[0], expected,
                                       rtol=4e-5, atol=2e-5)
            captured.append(list(values))
        ordinary = []
        for _ in range(5):
            _, value = owner.invoke(repeats=1024, timed=True)
            ordinary.append(value)
        public = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(25):
                owner.update(arrays)
                generation, _ = owner.invoke()
                result = owner.read(generation)[0]
            public.append((time.perf_counter() - start) * 1e3 / 25)
        np.testing.assert_allclose(result, expected, rtol=4e-5, atol=2e-5)
    return {
        "shape_mnk": list(shape), "scale_format": "e8m0",
        "route": "textual Graph -> native Schedule -> Tile -> GPU Target -> native image",
        "correctness": "passed_before_and_after_timing", "max_abs_error": error,
        "compile_and_package_ms": compile_ms,
        "members": ["scaled_product", "result_permutation"],
        "captured_repeats": repeats, "captured_member_samples_ms": captured,
        "captured_member_median_ms": [median(row[i] for row in captured) for i in range(2)],
        "captured_min_window_ms": min(min(row) * repeats for row in captured),
        "ordinary_program_repeats": 1024, "ordinary_program_samples_ms": ordinary,
        "ordinary_program_median_ms": median(ordinary),
        "upload_execute_readback_samples_ms": public,
        "upload_execute_readback_median_ms": median(public),
        "timing_scope": {
            "captured": "grouped pure SSA members; includes device graph dispatch; excludes capture, instantiation and copies",
            "ordinary": "interleaved native program HIP event window, includes native enqueue gaps",
            "end_to_end": "prepared owner update/upload, native invocation and copied readback; excludes compilation",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    architecture = runtime._rocm_live_arch()
    if architecture != "gfx1201":
        raise RuntimeError(f"owning gfx1201 required, observed {architecture}")
    library = os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]
    root = Path(__file__).resolve().parents[2]
    paths = [Path(os.environ["TESSERA_OPT"]), Path(os.environ["TESSERA_ROCM_OPT"]),
             Path(library), root / "python/tessera/compiler/native_scaled_program.py",
             root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp"]
    packet = {
        "architecture": architecture,
        "device_inventory": subprocess.run(["rocminfo"], check=True, capture_output=True,
                                           text=True, timeout=30).stdout,
        "identity_sha256": {str(path.relative_to(root) if path.is_relative_to(root) else path):
                            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases": [record(shape, library) for shape in
                  [(17, 19, 64), (31, 47, 95), (64, 65, 128)]],
        "claim": "bounded textual native package correctness and cost attribution; no public mapped-output or speedup closure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
