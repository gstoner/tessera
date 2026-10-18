"""Correctness-gated owning-device native source-pullback characterization.

Public warm, prepared and portable package calls are measured separately.
No timing domain is promoted as isolated kernel latency or a speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
import time

from tessera import runtime as rt
from tessera.compiler.native_rocm_source_reverse import package_source_reverse, RocmSourceReverseProgram
from benchmarks.math.source_reverse_reference import inputs, check
from tests.unit.test_native_rocm_source_reverse import source

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "python/tessera/compiler/native_rocm_source_reverse.py",
    "python/tessera/compiler/rocm_source_reverse_runtime.py",
    "python/tessera/compiler/native_vjp_composites.py",
    "python/tessera/compiler/native_vjp_plugins.py",
    "python/tessera/compiler/execution_matrix.py",
    "python/tessera/compiler/jit.py",
    "python/tessera/runtime.py",
    "python/tessera/compiler/native_sm120_source_reverse.py",
    "python/tessera/compiler/native_scaled_program.py",
    "python/tessera/compiler/rocm_scalar_matmul.py",
    "python/tessera/compiler/rocm_pipeline.py",
    "python/tessera/compiler/pipeline_registry.py",
    "benchmarks/math/source_reverse_reference.py",
    "benchmarks/nvidia/active_source_jvp_reference.py",
    "tests/unit/test_native_rocm_source_reverse.py",
    "benchmarks/rocm/record_native_source_reverse.py",
)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def record(architecture, output, *, samples=5, repeats=128):
    if (architecture not in {"gfx1151", "gfx1201"} or
            any(type(v) is not int or v <= 0 for v in (samples, repeats)) or repeats > 65536):
        raise ValueError("owning architecture and positive bounded sampling required")
    if rt._rocm_live_arch() != architecture:
        raise RuntimeError("source pullback recorder requires its exact owning GPU")
    info = subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True)
    device = re.search(r"Name:\s+" + architecture + r"\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)", info)
    if device is None:
        raise RuntimeError("owning device UUID/name unavailable")
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    from tessera.compiler.rocm_native import _linked_tool_files
    compiler = find_tessera_opt()
    if compiler is None:
        raise RuntimeError("matching compiler unavailable")
    compiler_files = {str(path): sha(path.read_bytes())
                      for path in (compiler.resolve(), *_linked_tool_files(compiler))}
    source_hashes = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    rows = []
    for storage in ("f16", "bf16"):
        for permuted in (False, True):
            start = time.perf_counter_ns()
            graph = source(architecture, storage, permuted)
            capture_ms = (time.perf_counter_ns() - start)/1e6
            start = time.perf_counter_ns()
            package = package_source_reverse(graph)
            package_ms = (time.perf_counter_ns() - start)/1e6
            serialized = package.to_json()
            values = inputs(storage, permuted)
            start = time.perf_counter_ns()
            owner = package.prepare(values)
            prepare_ms = (time.perf_counter_ns() - start)/1e6
            try:
                generation, _ = owner.invoke()
                check(owner.read(generation), values, permuted)
                provider_path = Path(owner.lib._name)
                if not provider_path.is_file():
                    raise RuntimeError("native provider fingerprint unavailable")
                stages, events, prepared, portable = [], [], [], []
                for window in range(samples):
                    values = inputs(storage, permuted, 83 + 211*window)
                    owner.update(values)
                    generation, stage_ms = owner.profile_members(repeats=repeats)
                    check(owner.read(generation), values, permuted)
                    stages.append(stage_ms)
                    generation, event_ms = owner.invoke(repeats=repeats, timed=True)
                    check(owner.read(generation), values, permuted)
                    if not all(math.isfinite(v) and v > 0 for v in (*stage_ms, event_ms)):
                        raise RuntimeError("native timing regions are unavailable")
                    events.append(event_ms)
                    start = time.perf_counter_ns()
                    owner.update(values)
                    generation, _ = owner.invoke()
                    actual = owner.read(generation)
                    prepared.append((time.perf_counter_ns() - start)/1e6)
                    check(actual, values, permuted)
                    start = time.perf_counter_ns()
                    restored = RocmSourceReverseProgram.from_json(serialized)
                    with restored.prepare(values) as temporary:
                        generation, _ = temporary.invoke()
                        actual = temporary.read(generation)
                    portable.append((time.perf_counter_ns() - start)/1e6)
                    check(actual, values, permuted)
                import tessera as ts
                from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed
                from unittest.mock import patch
                fn = ts.jit(target="rocm_" + architecture, autodiff="reverse", wrt=("a",))(
                    product_reversed if permuted else product)
                public = []
                try:
                    start = time.perf_counter_ns()
                    actual = fn.native_backward(*values[:2], out_cotangents=values[2])
                    public_first_ms = (time.perf_counter_ns() - start)/1e6
                    check(actual, values, permuted)
                    for window in range(samples):
                        values = inputs(storage, permuted, 1151 + 211*window)
                        with patch("subprocess.run", side_effect=RuntimeError("warm public VJP requested compiler")):
                            start = time.perf_counter_ns()
                            actual = fn.native_backward(*values[:2], out_cotangents=values[2])
                            public.append((time.perf_counter_ns() - start)/1e6)
                        check(actual, values, permuted)
                    public_certificate = fn.last_backward_execution["execution_certificate"]
                    if public_certificate["evidence_scope"] != "exact_device":
                        raise RuntimeError("public VJP lacks owning physical evidence")
                finally:
                    fn.close_native_storage()
                row = {
                    "primal_shape_mnk": [17, 19, 35], "storage": storage, "permuted_roots": permuted,
                    "correctness": "independent_f64_bound_before_and_during_timing",
                    "frontend_capture_ms": capture_ms, "compiler_package_ms": package_ms,
                    "cold_owner_prepare_ms": prepare_ms,
                    "public_first_call_after_image_priming_ms": public_first_ms,
                    "public_warm_completed_samples_ms": public,
                    "public_warm_completed_median_ms": median(public),
                    "public_execution_certificate": public_certificate,
                    "captured_member_samples_ms": stages,
                    "captured_member_medians_ms": [median(v[i] for v in stages) for i in range(2)],
                    "native_program_event_samples_ms": events, "native_program_event_median_ms": median(events),
                    "prepared_update_invoke_read_samples_ms": prepared,
                    "prepared_update_invoke_read_median_ms": median(prepared),
                    "completed_portable_call_samples_ms": portable, "completed_portable_call_median_ms": median(portable),
                    "compiler_cache_identity": package.compiler_identity,
                    "compiler_binary_sha256": compiler_files[str(compiler.resolve())],
                    "provider_sha256": sha(provider_path.read_bytes()),
                    "payload_sha256": [sha(image) for image in package.images],
                    "package_sha256": sha(serialized), "native_program_digest": package.program_digest,
                    "entries": [package.scalar_descriptor.entry_symbol, package.frames[1]["entry"]],
                    "frames": [{"abi": frame["abi"], "scalars": frame["scalars"], "geometry": frame["geometry"]}
                               for frame in package.frames],
                    "member_stage_digests": package.stage_digests,
                }
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("storage", "permuted_roots", "captured_member_medians_ms",
                    "native_program_event_median_ms", "prepared_update_invoke_read_median_ms",
                    "completed_portable_call_median_ms", "public_warm_completed_median_ms")}), flush=True)
            finally:
                owner.close()
    if source_hashes != {name: sha((ROOT / name).read_bytes()) for name in SOURCES}:
        raise RuntimeError("source changed during source-pullback recording")
    result = {
        "schema": "tessera.rocm.source_reverse_characterization.v1",
        "architecture": architecture, "device_uuid": device[1], "device_name": device[2].strip(),
        "scope": "named static public/prepared/portable host-root source VJP",
        "route": "frontend Graph->paired AD->native SSA members->Schedule->Tile->ROCm Target->LLVM/HSACO->checked native owner",
        "compiler_files_sha256": compiler_files,
        "source_sha256": source_hashes, "samples": samples, "repeats": repeats,
        "timing_domains": {
            "public_warm_completed": "ordinary certified native_backward with retained native owner; first call follows image priming",
            "captured_member": "device graph member dispatch; capture/instantiate outside region",
            "native_program_event": "native stream launch-loop event window, already divided by repeats; may include enqueue gaps",
            "prepared_update_invoke_read": "completed host root refresh plus native invoke/read",
            "completed_portable_call": "completed deserialize/validate/prepare/invoke/read/close host call",
        },
        "rows": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gfx1151", "gfx1201"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=128)
    args = parser.parse_args()
    record(args.architecture, args.output, samples=args.samples, repeats=args.repeats)
