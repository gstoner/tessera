#!/usr/bin/env python3
"""Matched logical input evaluation of gfx1201 FP8, MXFP8 and folded MXFP4.

This compares existing compiler-owned package strategies, not isolated dtype
costs: FP8 uses K128/N128, MXFP8 uses K32/N1 and folded MXFP4 uses BM256/BN64.
An FP8 K32/N1 control exposes granularity dependence. Native selectors still
choose different panels, so it does not isolate the scale-consumer cost.
The approximate folded physical route stores expanded E4M3 weights, not packed
four-bit weights. Quantization and folding losses are reported independently.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    FoldedPrefillSchedule, folded_prefill_grid, prepare_folded_weights,
)
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul
from tessera.compiler.rocm_mxfp4_packed_folded import (
    prepare_packed_folded_payload, compile_packed_folded_scaled_matmul,
)
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from tests._support import rocm_isa


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def errors(got, expected):
    delta = np.asarray(got, dtype=np.float64) - expected
    norm = float(np.linalg.norm(expected))
    return dict(max_abs=float(np.max(np.abs(delta))),
                relative_rms=float(np.linalg.norm(delta) / max(norm, np.finfo(float).tiny)))


def quantize_fp8(a, b, group_k, group_n, *, e8m0=False):
    m, k = a.shape
    n = b.shape[1]
    av = a.reshape(m, k // group_k, group_k)
    # Pad only the quantization group; logical B remains [K,N].
    padded = np.pad(b, ((0, 0), (0, (-n) % group_n)))
    bv = padded.reshape(k // group_k, group_k, padded.shape[1] // group_n, group_n)
    amax = np.max(np.abs(av), axis=2)
    bmax = np.max(np.abs(bv), axis=(1, 3))
    if e8m0:
        ae = np.ceil(np.log2(np.maximum(amax, np.finfo(np.float32).tiny) / 448)).astype(np.int16)
        be = np.ceil(np.log2(np.maximum(bmax, np.finfo(np.float32).tiny) / 448)).astype(np.int16)
        ae, be = np.maximum(ae, -127), np.maximum(be, -127)
        if np.any(ae > 127) or np.any(be > 127):
            raise ValueError("input outside finite standard E8M0 scale range")
        sa, sb = np.exp2(ae).astype(np.float32), np.exp2(be).astype(np.float32)
        encoded = (np.ascontiguousarray((ae + 127).astype(np.uint8)),
                   np.ascontiguousarray((be + 127).astype(np.uint8)))
    else:
        sa = np.maximum(amax / 448, np.finfo(np.float32).tiny).astype(np.float32)
        sb = np.maximum(bmax / 448, np.finfo(np.float32).tiny).astype(np.float32)
        encoded = (np.ascontiguousarray(sa), np.ascontiguousarray(sb))
    qa = np.ascontiguousarray((av / sa[..., None]).astype(ml_dtypes.float8_e4m3fn).reshape(m, k))
    qb = np.ascontiguousarray((bv / sb[:, None, :, None]).astype(ml_dtypes.float8_e4m3fn)
                             .reshape(k, -1)[:, :n])
    da = qa.astype(np.float64) * np.repeat(sa.astype(np.float64), group_k, axis=1)
    db = qb.astype(np.float64) * np.repeat(
        np.repeat(sb.astype(np.float64), group_k, axis=0), group_n, axis=1)[:, :n]
    return qa, qb, encoded, da, db


def _quantize_fp4_rows(b, *, rows_per_batch=64):
    """Bound distance scratch while preserving E2M1 nearest/even decisions."""
    k,n=b.shape
    if k<=0 or n<=0 or k%32 or rows_per_batch<=0:
        raise ValueError("FP4 quantization requires K32 blocks and positive row batches")
    codes=np.empty((n,k),np.uint8)
    exponents=np.empty((n,k//32),np.int16)
    magnitudes=np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    priority=np.where(np.arange(8)%2==0,np.arange(8),np.arange(8)+8)
    for start in range(0,n,rows_per_batch):
        stop=min(start+rows_per_batch,n)
        blocks=b[:,start:stop].T.reshape(stop-start,k//32,32)
        exp=np.ceil(np.log2(np.maximum(np.max(np.abs(blocks),axis=2),
                                      np.finfo(np.float32).tiny)/6)).astype(np.int16)
        exp=np.maximum(exp,-126)
        if np.any(exp>127):
            raise ValueError("input outside folded finite nonzero E8M0 scale range")
        normalized=blocks/np.exp2(exp)[...,None]
        distances=np.abs(np.abs(normalized)[...,None]-magnitudes)
        nearest=distances.min(axis=-1,keepdims=True)
        indices=np.argmin(np.where(distances==nearest,priority,100),axis=-1)
        codes[start:stop]=(indices+8*np.signbit(normalized)).astype(np.uint8).reshape(stop-start,k)
        exponents[start:stop]=exp
    return codes,np.ascontiguousarray((exponents.T+127).astype(np.uint8))


def quantize_folded(a,b,*,packed_weights=None):
    m,k=a.shape
    n=b.shape[1]
    sa=np.maximum(np.max(np.abs(a),axis=1)/448,np.finfo(np.float32).tiny).astype(np.float32)
    qa=np.ascontiguousarray((a/sa[:,None]).astype(ml_dtypes.float8_e4m3fn).view(np.uint8))
    if packed_weights is None:
        codes,scales=_quantize_fp4_rows(b)
    else:
        packed,scales=packed_weights
        codes=mx.unpack_e2m1_codes(packed)
        if codes.shape!=(n,k) or scales.shape!=(k//32,n):
            raise ValueError("checkpoint FP4 source shape differs from matched operands")
    folded=prepare_folded_weights(mx.pack_e2m1_codes(codes),scales,allow_approximate=True)
    da=qa.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*sa.astype(np.float64)[:,None]
    magnitudes=np.array([0,.5,1,1.5,2,3,4,6],np.float64)
    # Decode independently from the folding implementation; this named
    # physical profile uses code zero as a zero scale, not standard E8M0.
    factors=np.where(scales==0,0.,np.exp2(scales.astype(np.int16)-127)).T
    exact_b=(magnitudes[codes&7]*np.where(codes&8,-1.,1.)*
             np.repeat(factors,32,axis=1)).T
    folded_b=(folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*
              np.exp2(folded.row_reference.astype(np.int16)-127)[...,None]).T
    return qa,sa,folded,da,exact_b,folded_b


def verify(output, ideal, abs_product, k):
    got = output.astype(np.float64)
    if not np.isfinite(got).all():
        raise AssertionError("nonfinite package output")
    # Conservative f32 accumulation/scaling forward bound plus BF16 store
    # rounding, checked elementwise against independently decoded f64 operands.
    gamma = (k + 8 * (k // 32) + 16) * 2.**-24
    gamma /= 1 - gamma
    bound = gamma * abs_product + 2.**-8 * (np.abs(ideal) + gamma * abs_product) + 2.**-126
    delta = np.abs(got - ideal)
    if np.any(delta > bound):
        index = np.unravel_index(np.argmax(delta - bound), ideal.shape)
        raise AssertionError(f"native arithmetic exceeds forward bound at {index}: {delta[index]} > {bound[index]}")
    return dict(errors(got, ideal), correctness="elementwise_f32_forward_bound_plus_bf16_store",
                max_error_bound=float(bound.max()), violations=0)


def run(shape, args, hip, clock):
    m, n, k = shape
    rng = np.random.default_rng(args.seed + m + n + k)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    # Nonuniform block magnitudes exercise actual MXFP4 folding losses.
    b *= np.repeat(np.exp2(rng.integers(-6, 7, (k // 32, n))).astype(np.float32), 32, axis=0)
    return run_operands(a,b,args,hip,clock)


def run_operands(a,b,args,hip,clock,*,packed_weights=None):
    """Evaluate actual matched arrays; synthetic generation stays in run()."""
    a=np.asarray(a,dtype=np.float32)
    b=np.asarray(b,dtype=np.float32)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError("matched operands must be compatible rank-two arrays")
    m,k=a.shape
    n=b.shape[1]
    shape=(m,n,k)
    if m<=64 or n<=0 or k<=0 or k%128:
        raise ValueError("existing folded comparison requires M>64 and K divisible by128")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("matched source arrays must be finite")
    packed_weights={} if packed_weights is None else packed_weights
    if any(not name.startswith("mxfp4_") or name=="mxfp4_folded" for name in packed_weights):
        raise ValueError("checkpoint arms must use distinct mxfp4_ names")
    original = a.astype(np.float64) @ b.astype(np.float64)
    records, engines, bindings = {}, {}, {}
    graphs = FoldedGraphWindows(clock)
    try:
        names = ["fp8_k128_n128", "fp8_k32_n1_control", "mxfp8_k32_n1", "mxfp4_folded", *packed_weights]
        if getattr(args, "include_mxfp8_k64", False):
            names += ["mxfp8_lds_k32_control", "mxfp8_lds_k64_candidate"]
        if getattr(args, "include_mxfp8_seed", False):
            names += ["mxfp8_seed_control"]
        if getattr(args, "include_native_packed", False):
            if n % 16:
                raise ValueError("native packed comparison requires N divisible by16")
            names += [name + "_native_packed" for name in names if name.startswith("mxfp4_")]
        if getattr(args, "reverse_arms", False):
            names.reverse()
        for name in names:
            output = np.full((m, n), -101, ml_dtypes.bfloat16)
            start = time.perf_counter_ns()
            extra, geometry = {}, {}
            if name.startswith("mxfp4_"):
                packed_arm = name.endswith("_native_packed")
                base_name = name.removesuffix("_native_packed")
                source_weights = packed_weights.get(base_name)
                qa, sa, folded, da, exact_b, db = quantize_folded(a, b, packed_weights=source_weights)
                package = compile_folded_scaled_matmul(
                    qa, sa, folded, tessera_opt=args.compiler, allow_approximate=True).package
                prov = package.descriptor.provenance
                schedule = FoldedPrefillSchedule(
                    raster_group_m=prov["raster_group_m"], workgroup_mode=prov["workgroup_mode"],
                    staging_prefetch=prov["staging_prefetch"], epilogue=prov["epilogue_schedule"],
                    row_guard=prov["row_guard"])
                geometry = dict(grid=folded_prefill_grid(m, n, schedule), block=(256, 1, 1))
                buffers = dict(a=qa, b_folded=folded.weight_bytes, a_scale=sa,
                               row_reference=folded.row_reference, output=output)
                if packed_arm:
                    if source_weights is None:
                        codes, scales = _quantize_fp4_rows(b)
                        source_weights = (mx.pack_e2m1_codes(codes), scales)
                    payload = prepare_packed_folded_payload(*source_weights, allow_approximate=True)
                    package = compile_packed_folded_scaled_matmul(
                        qa, sa, payload, tessera_opt=args.compiler).package
                    geometry = dict(grid=((n + 63)//64, (m + 255)//256, 1), block=(256, 1, 1))
                    buffers = dict(a=qa, b_packed=payload.weight_bytes, a_scale=sa,
                                   scale_plane=payload.scale_plane, output=output)
                extra = dict(approximate_policy="explicit_allow", fold_lossless=folded.lossless,
                    fold_inexact_value_count=folded.inexact_value_count,
                    folding_weight_error=errors(db, exact_b),
                    folding_output_error=errors(da @ db, da @ exact_b),
                    pre_fold_quantized_output_error=errors(da @ exact_b, original),
                    physical_weight_storage="packed_e2m1_fragment_bytes" if packed_arm else "expanded_e4m3_bytes")
            else:
                gk, gn = (128, 128) if name == "fp8_k128_n128" else (32, 1)
                qa, qb, scales, da, db = quantize_fp8(a, b, gk, gn, e8m0=name.startswith("mxfp8"))
                profile = BlockScaleShape(m, n, k, gk, gn, "nk", "bf16")
                policy = ("seed" if name == "mxfp8_seed_control" else
                          "lds_k64" if name == "mxfp8_lds_k64_candidate" else
                          "lds" if name == "mxfp8_lds_k32_control" else "auto")
                package = (compile_mxfp8(profile, schedule_policy=policy)
                           if name.startswith("mxfp8") else compile_blockscale(profile))
                buffers = dict(a=qa, b=np.ascontiguousarray(qb.T), a_scale=scales[0],
                               b_scale=scales[1], o=output)
                extra = dict(group_k=gk, group_n=gn, approximate_policy="none",
                             physical_weight_storage="e4m3_bytes")
            prepare_compile_ms = (time.perf_counter_ns() - start) / 1e6
            ideal = da @ db
            abs_product = np.abs(da) @ np.abs(db)
            artifact = rt.RuntimeArtifact(metadata={"target": package.image.target},
                native_image=package.image, launch_descriptor=package.descriptor,
                tile_ir=package.tile_ir, target_ir=package.target_ir)
            binding = dict(buffers=buffers, scalars=dict(M=m, N=n, K=k))
            receipt = rt.launch(artifact, binding)
            if not receipt.get("ok") or receipt.get("execution_kind") != "native_gpu":
                raise RuntimeError(receipt)
            correctness = verify(output, ideal, abs_product, k)
            profile = BlockScaleShape(m, n, k, 32, 1, "nk", "bf16")
            engine = Resident(hip, package, buffers, profile, **geometry)
            engines[name] = engine
            engine.launch_on_stream(None)
            verify(engine.result(output), ideal, abs_product, k)
            bindings[name] = (ideal, abs_product, output)
            isa = rocm_isa.disassemble(package.image.payload, chip="gfx1201")
            if "v_wmma_f32_16x16x16_fp8_fp8" not in isa:
                raise RuntimeError("no gfx1201 FP8 WMMA in native image")
            isa_file = f"{args.output.stem}_{m}_{n}_{k}_{name}.s"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / isa_file).write_text(isa)
            e2e = []
            for _ in range(args.windows):
                begin = time.perf_counter_ns()
                for _ in range(3):
                    receipt = rt.launch(artifact, binding)
                    if not receipt.get("ok"):
                        raise RuntimeError(receipt)
                e2e.append((time.perf_counter_ns() - begin) / 1e6 / 3)
            verify(output, ideal, abs_product, k)
            records[name] = dict(extra, correctness=correctness,
                quantized_weight_error=errors(db,b),
                quantized_ideal_output_error=errors(ideal, original),
                native_output_error=errors(output.astype(np.float64), original),
                prepare_compile_ms=prepare_compile_ms, abi=package.descriptor.abi_id,
                native_provenance=dict(package.descriptor.provenance),
                payload_sha256=hashlib.sha256(package.image.payload).hexdigest(),
                tile_sha256=hashlib.sha256(package.tile_ir.encode()).hexdigest(),
                target_sha256=hashlib.sha256(package.target_ir.encode()).hexdigest(),
                isa_file=isa_file, isa_sha256=hashlib.sha256(isa.encode()).hexdigest(),
                static_isa_counts=dict(rocm_isa.mnemonics(isa, r"(?:v_(?:wmma|mul|cvt|add|ldexp)_\w+|ds_(?:load|store|read|write)\w*|s_barrier\w*)")),
                grid=engine.grid, workgroup=engine.block,
                route=package.descriptor.provenance["route"],
                schedule={key: value for key, value in package.descriptor.provenance.items()
                          if key in ("staging", "warps", "pipeline_depth", "macro_tile", "macro_k",
                                     "scale_k", "scale_n", "scale_format", "block_m", "block_n",
                                     "block_k", "tile_m_per_wave", "tile_n_per_wave",
                                     "raster_group_m", "staging_prefetch", "row_guard",
                                     "epilogue_schedule", "image_shape_policy")},
                physical_input_bytes=sum(v.nbytes for key, v in buffers.items() if v is not output),
                end_to_end_ms=e2e, end_to_end_median_ms=statistics.median(e2e), device_windows=[])
        counts = {}
        for name, engine in engines.items():
            count = 32
            while graphs.window(engine, count, bracketed=True)["device_window_ms"] < 20:
                count *= 2
            counts[name] = count
        for trial in range(args.windows):
            order = list(engines)
            # Rotate then reverse to balance which arm follows which, avoiding
            # always keeping the folded arm at the end of the thermal sequence.
            order = order[trial % len(order):] + order[:trial % len(order)]
            if trial % 2:
                order.reverse()
            for name in order:
                sample = graphs.window(engines[name], counts[name], bracketed=True)
                if sample["device_event_disagreement"] > .05:
                    raise RuntimeError(f"device timer disagreement: {sample}")
                records[name]["device_windows"].append(sample)
        for name, engine in engines.items():
            ideal, abs_product, output = bindings[name]
            verify(engine.result(output), ideal, abs_product, k)
            records[name]["device_execution_dispatch_median_ms"] = statistics.median(
                s["device_window_ms"] / s["launches"] for s in records[name]["device_windows"])
        return dict(shape_mnk=list(shape), input_sha256=dict(a=digest(a), b=digest(b)),
                    source_output_norm=float(np.linalg.norm(original)), arms=records)
    finally:
        graphs.close()
        for engine in engines.values():
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverse-arms", action="store_true",
        help="reverse construction/calibration and initial measurement order")
    parser.add_argument("--include-mxfp8-seed", action="store_true",
        help="retain the native one-wave seed as a matched control for automatic selector trials")
    parser.add_argument("--include-mxfp8-k64", action="store_true",
        help="compare explicit K64 LDS candidate with forced K32 LDS on identical operands")
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source-dir", type=Path,
        help="saved native PM/generator/identity sources matching a preserved reference compiler")
    parser.add_argument("--include-native-packed", action="store_true",
        help="add native packed E2M1 arms matched to each expanded MXFP4 source")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--shapes", default="200,256,128;200,256,1024;200,256,1536;200,256,2048;256,1024,1024;256,4096,5120")
    args = parser.parse_args()
    shapes = [tuple(map(int, item.split(","))) for item in args.shapes.split(";")]
    if args.windows < 3 or any(len(s) != 3 or s[0] <= 64 or s[1] <= 0 or s[2] <= 0 or s[2] % 128 for s in shapes):
        parser.error("at least three windows and M>64, N>0, K divisible by 128 required")
    os.environ["TESSERA_OPT"] = str(args.compiler.resolve())
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires a live gfx1201 GPU")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(name, len(name), ordinal.value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    packet = dict(schema="tessera.gfx1201.three_format_package_evaluation.v1",
        architecture="gfx1201", device=name.value.decode(), device_ordinal=ordinal.value,
        seed=args.seed, compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        timing_scope="resident graph device execution plus GPU dispatch; checked runtime staging/transfers/sync separately",
        selector_promotion=False, public_dtype_promotion=False,
        comparison_scope="matched f32 source operands; existing distinct schedules/scales and approximate folded physical contract",
        rows=[])
    sources = [__file__, ROOT / "benchmarks/rocm/benchmark_gfx1201_mxfp8_package.py",
               ROOT / "python/tessera/compiler/rocm_fp8_blockscale.py",
               ROOT / "python/tessera/compiler/rocm_mxfp8_blockscale.py",
               ROOT / "python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
               ROOT / "python/tessera/compiler/rocm_mxfp4.py",
               ROOT / "python/tessera/compiler/rocm_mxfp4_packed_folded.py",
               ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMFoldedW4A8Contract.h",
               ROOT / "python/tessera/runtime.py",
               ROOT / "python/tessera/compiler/rocm_mxfp4_folded_carrier.py",
               ROOT / "src/compiler/programming_model/lib/PMPasses.cpp",
               ROOT / "src/compiler/programming_model/ir/ScheduleDialect.cpp",
               ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
               ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
               ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp"]
    packet["source_sha256"] = {str(Path(p).relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if args.reference_source_dir:
        for file in sources:
            file = Path(file)
            if file.name in {"PMPasses.cpp", "ScheduleDialect.cpp", "GenerateWMMAGemmKernel.cpp", "ROCMKernelIdentity.cpp"}:
                saved = args.reference_source_dir / file.name
                packet["source_sha256"][str(file.relative_to(ROOT))] = hashlib.sha256(saved.read_bytes()).hexdigest()
        packet["reference_source_override"] = str(args.reference_source_dir)
    try:
        for shape in shapes:
            row = run(shape, args, hip, clock)
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet, indent=2) + "\n")
            print(json.dumps(dict(shape=shape, arms={n: dict(
                device_us=r["device_execution_dispatch_median_ms"] * 1000,
                quality_relative_rms=r["native_output_error"]["relative_rms"])
                for n, r in row["arms"].items()})), flush=True)
    finally:
        clock.close()


if __name__ == "__main__":
    main()
