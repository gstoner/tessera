"""Public gfx1201 expansion proof and separated cold/host/device timings."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import lzma
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import nvfp4_expansion as reference
from benchmarks.rocm.benchmark_nvfp4_native_expansion import ResidentExpansion
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from tests._support import rocm_isa


def make_public(offsets):
    @ts.jit(target="rocm_gfx1201")
    def expand(codes, scales, globals_):
        return ts.ops.nvfp4_expand(codes, scales, globals_,
            row_offsets=offsets, numeric_policy=reference.nvfp4_expansion_policy())
    return expand


def inputs(checkpoint):
    if checkpoint:
        from benchmarks.rocm.benchmark_rocm_nvfp4_checkpoint import _load_projection
        projections, sources = [], []
        for role in ("gate_proj", "up_proj"):
            item = _load_projection("model.layers.0.mlp." + role + ".weight")
            projections.append(item["projection"])
            sources.append(item["source"])
        offsets = [0]
        for p in projections:
            offsets.append(offsets[-1] + p.packed_codes.shape[0])
        yield (
            "pinned_checkpoint_gate_up",
            np.concatenate([p.packed_codes for p in projections]),
            np.concatenate([p.e4m3_scales for p in projections]),
            np.array([p.global_scale for p in projections], np.float64),
            offsets, sources)
    else:
        for n, k in ((7, 48), (128, 4096), (24576, 4096)):
            rng = np.random.default_rng(314)
            codes = rng.integers(0, 256, (n, k//2), dtype=np.uint8)
            scales = rng.integers(0, 127, (n, k//16), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
            scales[0, 0] = -0.
            yield ("seeded_synthetic", codes, scales,
                   np.array([.000123456789, .007654321], np.float64),
                   [0, n//2, n], [])


def measure(args):
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    hip = rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("HIP unavailable")
    ResidentExpansion.check(hip.hipInit(0))
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    ResidentExpansion.check(hip.hipGetDevice(C.byref(ordinal)))
    ResidentExpansion.check(hip.hipDeviceGetName(name, len(name), ordinal.value))
    if not name.value:
        raise RuntimeError("missing actual device identity")
    records = []
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    try:
        for kind, codes, scales, globals_, offsets, sources in inputs(args.checkpoint):
            expected = reference.reference_nvfp4_expand(
                codes, scales, globals_, row_offsets=offsets,
                numeric_policy=reference.nvfp4_expansion_policy())
            function = make_public(offsets)

            def compare(outputs):
                for got, want, dtype in zip(outputs, expected, (np.uint8, np.uint32)):
                    np.testing.assert_array_equal(got.view(dtype), want.view(dtype))

            oracle = reference.reference_nvfp4_expand
            def forbidden(*a, **kw):
                raise AssertionError("native public route called the eager oracle")
            reference.reference_nvfp4_expand = forbidden
            engine = graphs = None
            try:
                start = time.perf_counter_ns()
                got = function(codes, scales, globals_)
                cold_ms = (time.perf_counter_ns()-start)/1e6
                compare(got)
                artifact = function.runtime_artifact()
                if function.execution_kind != "native_gpu" or not artifact.metadata["canonical_executable"]:
                    raise AssertionError("public route did not execute a canonical native package")
                image = artifact.native_image
                descriptor = artifact.launch_descriptor
                warm_ms = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    got = function(globals_=globals_, scales=scales, codes=codes)
                    warm_ms.append((time.perf_counter_ns()-start)/1e6)
                    compare(got)
                    if function.runtime_artifact().native_image.image_digest != image.image_digest:
                        raise AssertionError("warm public call did not reuse the native image")
                restored = rt.RuntimeArtifact.from_json(artifact.to_json())
                arrays = [codes, scales, globals_, *[np.empty_like(x) for x in expected]]
                names = [b.name for b in sorted(descriptor.buffers, key=lambda b: b.ordinal)]
                replay_ms = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    receipt = rt.launch(restored, {"buffers": dict(zip(names, arrays)), "scalars": {}})
                    replay_ms.append((time.perf_counter_ns()-start)/1e6)
                    if not receipt.get("ok") or receipt["execution_kind"] != "native_gpu":
                        raise RuntimeError(receipt)
                    compare(arrays[3:])
                engine = ResidentExpansion(hip, SimpleNamespace(image=image, descriptor=descriptor), arrays)
                engine.launch_on_stream(None)
                compare(engine.outputs(arrays))
                graphs = FoldedGraphWindows(clock)
                count = 32
                while graphs.window(engine, count, bracketed=True)["device_window_ms"] < 20:
                    count *= 2
                    if count > 65536:
                        raise RuntimeError("unable to create stable timing window")
                windows = [graphs.window(engine, count, bracketed=True) for _ in range(3)]
                if any(w["device_event_disagreement"] > .05 for w in windows):
                    raise RuntimeError("device/event clocks disagree")
                compare(engine.outputs(arrays))
                isa = rocm_isa.disassemble(image.payload, chip="gfx1201")
                if "v_cvt_f32_f64" not in isa:
                    raise RuntimeError("native image missing declared f64-to-f32 rounding")
                isa_file = image.payload_digest + ".s.xz"
                (args.output.parent/isa_file).write_bytes(lzma.compress(isa.encode()))
                records.append({
                    "shape_nk": [codes.shape[0], 2*codes.shape[1]], "input_kind": kind,
                    "checkpoint_sources": sources, "row_offsets": offsets,
                    "projection_globals": globals_.tolist(),
                    "packed_input_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                    "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
                    "image_sha256": image.payload_digest, "descriptor_sha256": descriptor.descriptor_digest,
                    "schedule_sha256": hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
                    "tile_sha256": hashlib.sha256(artifact.tile_ir.encode()).hexdigest(),
                    "isa_file": isa_file, "isa_text_sha256": hashlib.sha256(isa.encode()).hexdigest(),
                    "correctness": "bitwise_both_outputs_cold_warm_serialized_and_resident",
                    "eager_oracle_during_execution": "forbidden",
                    "cold_first_public_call_ms": cold_ms, "warm_public_samples_ms": warm_ms,
                    "warm_public_median_ms": statistics.median(warm_ms),
                    "serialized_completed_host_samples_ms": replay_ms,
                    "serialized_completed_host_median_ms": statistics.median(replay_ms),
                    "device_graph_median_ms": statistics.median(w["device_window_ms"]/count for w in windows),
                    "device_graph_windows": windows,
                    "scope": "standalone producer; device graph excludes all host validation/staging/transfers; warm repeated inputs"})
                print(json.dumps(records[-1]), flush=True)
            finally:
                if graphs is not None:
                    graphs.close()
                if engine is not None:
                    engine.close()
                reference.reference_nvfp4_expand = oracle
    finally:
        clock.close()
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__), root/"python/tessera/compiler/nvfp4_expansion.py",
        root/"python/tessera/compiler/rocm_nvfp4_expand_native.py",
        root/"python/tessera/compiler/driver.py", root/"python/tessera/compiler/jit.py",
        root/"python/tessera/runtime.py", root/"python/tessera/compiler/backend_manifest.py",
        root/"python/tessera/compiler/capabilities.py"]
    return {"architecture": "gfx1201", "device": name.value.decode(),
            "route": "Python frontend->GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "compiler_sha256": hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            "records": records,
            "open": "resident consumer composition, whole-sequence native owner and model quality acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(measure(args), indent=2) + "\n")
