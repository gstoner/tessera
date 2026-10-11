"""Paired gfx1201 native NVFP4 winner-code rematerialization experiment."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_program import program_from_manifest
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
from benchmarks.rocm.record_bounded_nvfp4_rows import check_conversion, check_output, check_producer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_arm(n, k, destination):
    arguments = inputs_and_oracle(17, n, k)[0]
    function = make_function(n, k)
    program = function.compile_native_nvfp4_program(*arguments, m_bound=257)
    destination.write_text(json.dumps(program.manifest()) + "\n")


def profile(n, k, compilers):
    programs = {}
    compile_ms = {}
    with tempfile.TemporaryDirectory(prefix="nvfp4-remat-") as temporary:
        for arm, compiler in compilers.items():
            path = Path(temporary) / (arm + ".json")
            environment = dict(os.environ, TESSERA_OPT=compiler, TESSERA_ROCM_OPT=compiler)
            start = time.perf_counter()
            subprocess.run([sys.executable, __file__, "--compile-arm", str(n), str(k),
                            str(path)], env=environment, check=True)
            compile_ms[arm] = (time.perf_counter() - start) * 1000
            programs[arm] = program_from_manifest(json.loads(path.read_text()))
    records = []
    # The converter's weight envelope is N/K; changing M exercises the bounded
    # consumer and private storage lifetime without pretending converter work varies.
    for rows in (17, 257):
        arguments, _, converted, stored, expected = inputs_and_oracle(rows, n, k)
        sessions = {arm: program.native.native_session(*arguments, reuse=True)
                    for arm, program in programs.items()}
        try:
            checks = {}
            for arm, session in sessions.items():
                session.run_combined()
                checks[arm] = check_output(session.read_output(), expected)
                check_producer(session, converted, stored)
            timings = {arm: {"converter": [], "combined": []} for arm in sessions}
            public = {arm: [] for arm in sessions}
            windows = {arm: {} for arm in sessions}
            for round_index in range(7):
                order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
                diagnostics = {}
                for arm in order:
                    session = sessions[arm]
                    # Full rebinding invalidates ingest; alternate activation values.
                    updated = list(arguments)
                    factor = 1 if round_index % 2 == 0 else 2
                    updated[4] = arguments[4] * np.float32(factor)
                    round_expected = expected * factor
                    session.update_inputs(*updated)
                    for stage in ("converter", "combined"):
                        session.run_combined()
                        window = session.measure_graph(stage, samples=1, repeats=128)[0]
                        expected_nodes = 128 if stage == "converter" else 384
                        if window["graph_nodes"] != expected_nodes or window["host_graph_submissions"] != 1:
                            raise RuntimeError("captured geometry differs from declared stage")
                        timings[arm][stage].append(window["per_iteration_ms"])
                        windows[arm].setdefault(stage, []).append(window)
                        check_conversion(session, converted)
                        if stage == "combined":
                            check_producer(session, converted, stored)
                            checks[arm] = max(checks[arm], check_output(session.read_output(), round_expected))
                    diagnostics[arm] = session.conversion_diagnostics()
                    start = time.perf_counter()
                    output, receipts = programs[arm].execute(*updated)
                    public[arm].append((time.perf_counter() - start) * 1000)
                    if not all(receipt.get("ok") and receipt.get("execution_kind") == "native_gpu"
                               for receipt in receipts):
                        raise RuntimeError(receipts)
                    checks[arm] = max(checks[arm], check_output(output, round_expected))
                for name in ("packed", "exponents", "stats"):
                    np.testing.assert_array_equal(diagnostics["baseline"][name],
                                                  diagnostics["candidate"][name])
            records.append({
                "shape_mnk": [rows, n, k], "max_abs_output_error": checks,
                "correctness": "bitwise paired packed/exponents/stats; independent oracle at each stage",
                "event_samples_ms": timings, "captured_windows": windows,
                "event_median_ms": {arm: {stage: median(values) for stage, values in samples.items()}
                                    for arm, samples in timings.items()},
                "baseline_over_candidate": {stage: median(timings["baseline"][stage]) /
                                            median(timings["candidate"][stage])
                                            for stage in ("converter", "combined")},
                "package_end_to_end_samples_ms": public,
                "package_end_to_end_median_ms": {arm: median(samples) for arm, samples in public.items()},
            })
        finally:
            for session in sessions.values():
                session.close()
    return {"shape_nk": [n, k], "compile_process_ms": compile_ms,
            "image_digests": {arm: program.native.receipt["component_image_digests"]
                              for arm, program in programs.items()}, "frames": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-source", type=Path, required=False)
    parser.add_argument("--experiment", default="winner_code_rematerialization")
    parser.add_argument("--compile-arm", nargs=3)
    options = parser.parse_args()
    if options.compile_arm:
        n, k, destination = options.compile_arm
        compile_arm(int(n), int(k), Path(destination))
        return
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("exact gfx1201 required")
    if not options.baseline or not options.candidate or not options.output:
        parser.error("--baseline, --candidate and --output are required")
    active = subprocess.run(["pgrep", "-af", "[p]ytest|[g]raphify update"], capture_output=True, text=True)
    if active.returncode == 0 and active.stdout.strip():
        raise RuntimeError("test/graph jobs active")
    hip = ct.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = ct.create_string_buffer(256)
    uuid = (ct.c_ubyte * 16)()
    ordinal = ct.c_int()
    if (hip.hipInit(0) or hip.hipGetDevice(ct.byref(ordinal)) or
            hip.hipDeviceGetName(name, 256, ordinal.value) or
            hip.hipDeviceGetUuid(ct.byref(uuid), ordinal.value)):
        raise RuntimeError("live device query failed")
    compilers = {"baseline": str(options.baseline.resolve()), "candidate": str(options.candidate.resolve())}
    packet = {
        "schema": "tessera.gfx1201.nvfp4_winner_code_ab.v1",
        "experiment": options.experiment,
        "baseline_source_sha256": digest(options.baseline_source) if options.baseline_source else None,
        "source_sha256": {path: digest(path) for path in (
            "python/tessera/compiler/rocm_nvfp4_program.py",
            "python/tessera/compiler/native_nvfp4_program.py",
            "python/tessera/compiler/native_resident_nvfp4.py",
            "python/tessera/compiler/rocm_nvfp4_resident.py",
            "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
            "tests/device/rocm/test_native_nvfp4_ingest_leaf.py")},
        "architecture": "gfx1201", "device": {"name": name.value.decode(), "uuid_hex": bytes(uuid).hex()},
        "compiler_sha256": {arm: digest(path) for arm, path in compilers.items()},
        "runtime_sha256": digest(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
        "image_runtime_sha256": digest(os.environ["TESSERA_ROCM_NATIVE_IMAGE_LIB"]),
        "candidate_source_sha256": digest("src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/NativeNVFP4Ingest.h"),
        "recorder_sha256": digest(__file__),
        "route": "GraphIR->ScheduleIR->TileIR->ROCm Target IR->ROCDL/LLVM->HSACO",
        "timing_scope": "interleaved paired HIP captured windows; 128 launches; seven rounds; package wall time separate",
        "claim": "candidate experiment; no selector promotion; converter and combined timings are not summed",
        "profiles": [profile(n, k, compilers) for n, k in ((32, 64), (80, 256), (64, 1024), (256, 2048))],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
