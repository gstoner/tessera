"""Correctness-gated SM120 saved-LSE backward timing by timing domain."""
from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tessera import runtime as rt
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

SHAPE = (1, 4, 2, 16, 16, 32, 32)


def _types(shape=SHAPE):
    b, hq, hkv, sq, sk, d, dv = shape

    def tensor(*dims):
        encoded = "x".join(str(dim) for dim in dims)
        return IRType(f"tensor<{encoded}xf32>", tuple(str(dim) for dim in dims), "fp32")

    return tensor(b, hq, sq, d), tensor(b, hkv, sk, d), tensor(b, hkv, sk, dv), tensor(b, hq, sq, dv), tensor(b, hq, sq)


def _forward_module(shape=SHAPE):
    q, k, v, do, lse = _types(shape)
    b, hq, _, sq, _, _, dv = shape
    out = IRType(f"tensor<{b}x{hq}x{sq}x{dv}xf32>", tuple(map(str, (b, hq, sq, dv))), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_lse_forward", args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=[out, lse],
        body=[IROp(result="o,row_lse", op_name="tessera.flash_attn",
                   operands=["%q", "%k", "%v"], operand_types=[str(q), str(k), str(v)],
                   result_type=str(out), inferred_types=(out, lse), kwargs={"scale": shape[-2] ** -0.5, "causal": True, "lse_checkpoint": "saved"})],
        return_values=["%o", "%row_lse"],
    )])


def _backward_module(*, saved: bool, shape=SHAPE):
    q, k, v, do, lse = _types(shape)
    args = [IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%do", "%q", "%k", "%v"]
    kwargs = {"scale": shape[-2] ** -0.5, "causal": True, "route": "deterministic_direct",
              "deterministic": True, "workspace_limit_bytes": 0}
    if saved:
        args.extend((IRArg("output", do), IRArg("row_lse", lse)))
        operands.extend(("%output", "%row_lse"))
        kwargs["lse_checkpoint"] = "saved"
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_lse_backward", args=args, result_types=[q, k, v],
        body=[IROp(result="dq,dk,dv", op_name="tessera.flash_attn_bwd",
                   operands=operands, operand_types=[str(arg.ir_type) for arg in args],
                   result_type=f"({q}, {k}, {v})", kwargs=kwargs)],
        return_values=["%dq", "%dk", "%dv"],
    )])


def _compile(module):
    bundle = compile_graph_module(module, source_origin="NVIDIA-LSE-1", target="nvidia_sm120",
                                  options={"package_native": True}, enable_tool_validation=False)
    if bundle.native_image is None or bundle.launch_descriptor is None:
        raise RuntimeError("checkpoint graph did not produce a native package")
    return bundle, compile_result_from_bundle(bundle, module=module).to_runtime_artifact()


def _reference(q, k, v, do, scale):
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    dv = v.shape[-1]
    out = np.empty((b, hq, sq, dv), np.float32)
    lse = np.empty((b, hq, sq), np.float32)
    grads = [np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)]
    offset = max(sk - sq, 0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                visible = min(row + 1 + offset, sk)
                scores = scale * (k[batch, kv_head] @ q[batch, head, row])
                scores[visible:] = -np.inf
                row_lse = np.log(np.exp(scores - np.max(scores)).sum()) + np.max(scores)
                prob = np.exp(scores - row_lse)
                out[batch, head, row] = prob @ v[batch, kv_head]
                lse[batch, head, row] = row_lse
                delta = do[batch, head, row] @ out[batch, head, row]
                for key in range(visible):
                    ds = prob[key] * (do[batch, head, row] @ v[batch, kv_head, key] - delta)
                    grads[0][batch, head, row] += scale * ds * k[batch, kv_head, key]
                    grads[1][batch, kv_head, key] += scale * ds * q[batch, head, row]
                    grads[2][batch, kv_head, key] += prob[key] * do[batch, head, row]
    return out, lse, grads


def _cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def record(*, samples, warmup, device_reps, e2e_reps):
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("requires exact NVIDIA sm_120 device")
    shape = SHAPE
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(120_102)
    q = (rng.normal(size=(b, hq, sq, d)) * 0.2).astype(np.float32)
    k = (rng.normal(size=(b, hkv, sk, d)) * 0.2).astype(np.float32)
    v = (rng.normal(size=(b, hkv, sk, dv)) * 0.2).astype(np.float32)
    do = (rng.normal(size=(b, hq, sq, dv)) * 0.2).astype(np.float32)
    scale = d ** -0.5
    expected_o, expected_lse, expected_grads = _reference(q, k, v, do, scale)
    forward_module = _forward_module(shape)
    backward_module = _backward_module(saved=True, shape=shape)
    forward, forward_artifact = _compile(forward_module)
    backward, backward_artifact = _compile(backward_module)
    recompute, recompute_artifact = _compile(_backward_module(saved=False, shape=shape))
    dims = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), shape, strict=True))
    o, lse = np.empty_like(expected_o), np.empty_like(expected_lse)
    fwd = rt.launch(forward_artifact, {"q": q, "k": k, "v": v, "o": o, "row_lse": lse, **dims})
    if not fwd.get("ok") or fwd.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"saved-LSE forward did not execute natively: {fwd}")
    np.testing.assert_allclose(o, expected_o, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(lse, expected_lse, rtol=4e-5, atol=4e-5)
    saved_out = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    recompute_out = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    common = {"do": do, "q": q, "k": k, "v": v, **dims}
    saved = rt.launch(backward_artifact, {**common, "output": o, "row_lse": lse,
                     "dq": saved_out[0], "dk": saved_out[1], "dv": saved_out[2]})
    recomputed = rt.launch(recompute_artifact, {**common,
                          "dq": recompute_out[0], "dk": recompute_out[1], "dv": recompute_out[2]})
    if not saved.get("ok") or saved.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"saved-LSE backward did not execute natively: {saved}")
    if not recomputed.get("ok") or recomputed.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"recompute backward did not execute natively: {recomputed}")
    errors = []
    for actual, recomputed_value, expected in zip(saved_out, recompute_out, expected_grads, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)
        np.testing.assert_allclose(actual, recomputed_value, rtol=4e-5, atol=4e-5)
        errors.append(float(np.max(np.abs(actual - expected))))

    device_samples, e2e_samples = [], []
    forward_device_samples, forward_e2e_samples = [], []
    recompute_device_samples, recompute_e2e_samples = [], []
    with NvidiaDeviceSession() as session:
        qd, kd, vd, dod = (session.upload(x) for x in (q, k, v, do))
        od, lsed = session.empty(o.shape, np.float32), session.empty(lse.shape, np.float32)
        dqd, dkd, dvd = (session.empty(x.shape, np.float32) for x in (q, k, v))
        launch_fwd = rt.launch(forward_artifact,
                               {"q": qd, "k": kd, "v": vd, "o": od, "row_lse": lsed, **dims},
                               stream=session.stream)
        if not launch_fwd.get("ok") or launch_fwd.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"resident forward package failed: {launch_fwd}")
        np.testing.assert_allclose(session.download(od), expected_o, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(session.download(lsed), expected_lse, rtol=4e-5, atol=4e-5)
        forward_args = {"q": qd, "k": kd, "v": vd, "o": od, "row_lse": lsed, **dims}
        for _ in range(warmup):
            rt._nvidia_native_descriptor_resident_device_latency(
                forward.native_image, forward.launch_descriptor, forward_args,
                stream=session.stream, warmup=0, reps=1)
        for _ in range(samples):
            forward_device_samples.append(rt._nvidia_native_descriptor_resident_device_latency(
                forward.native_image, forward.launch_descriptor, forward_args,
                stream=session.stream, warmup=0, reps=device_reps))
        np.testing.assert_allclose(session.download(od), expected_o, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(session.download(lsed), expected_lse, rtol=4e-5, atol=4e-5)
        args = {"do": dod, "q": qd, "k": kd, "v": vd, "output": od, "row_lse": lsed,
                "dq": dqd, "dk": dkd, "dv": dvd, **dims}
        for _ in range(warmup):
            rt._nvidia_native_descriptor_resident_device_latency(
                backward.native_image, backward.launch_descriptor, args,
                stream=session.stream, warmup=0, reps=1)
        for _ in range(samples):
            device_samples.append(rt._nvidia_native_descriptor_resident_device_latency(
                backward.native_image, backward.launch_descriptor, args,
                stream=session.stream, warmup=0, reps=device_reps))
        for actual, expected in zip((session.download(x) for x in (dqd, dkd, dvd)),
                                    expected_grads, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)
        recompute_args = {key: value for key, value in args.items()
                          if key not in {"output", "row_lse"}}
        for _ in range(warmup):
            rt._nvidia_native_descriptor_resident_device_latency(
                recompute.native_image, recompute.launch_descriptor, recompute_args,
                stream=session.stream, warmup=0, reps=1)
        # Recompute is validated both before and after its event windows.
        for actual, expected in zip((session.download(x) for x in (dqd, dkd, dvd)),
                                    expected_grads, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)
        for _ in range(samples):
            recompute_device_samples.append(rt._nvidia_native_descriptor_resident_device_latency(
                recompute.native_image, recompute.launch_descriptor, recompute_args,
                stream=session.stream, warmup=0, reps=device_reps))
        resident_outputs = [session.download(x) for x in (dqd, dkd, dvd)]
        for actual, expected in zip(resident_outputs, expected_grads, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)

    for _ in range(samples):
        outs = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
        bindings = {**common, "output": o, "row_lse": lse, "dq": outs[0], "dk": outs[1], "dv": outs[2]}
        start = time.perf_counter()
        for _ in range(e2e_reps):
            result = rt.launch(backward_artifact, bindings)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"end-to-end backward launch failed: {result}")
        e2e_samples.append((time.perf_counter() - start) * 1e3 / e2e_reps)
        for actual, expected in zip(outs, expected_grads, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)

    for _ in range(samples):
        outs = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
        bindings = {**common, "dq": outs[0], "dk": outs[1], "dv": outs[2]}
        start = time.perf_counter()
        for _ in range(e2e_reps):
            result = rt.launch(recompute_artifact, bindings)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"end-to-end recompute launch failed: {result}")
        recompute_e2e_samples.append((time.perf_counter() - start) * 1e3 / e2e_reps)
        for actual, expected in zip(outs, expected_grads, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)

    for _ in range(samples):
        forward_o, forward_lse = np.empty_like(o), np.empty_like(lse)
        bindings = {"q": q, "k": k, "v": v, "o": forward_o, "row_lse": forward_lse, **dims}
        start = time.perf_counter()
        for _ in range(e2e_reps):
            result = rt.launch(forward_artifact, bindings)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"end-to-end forward launch failed: {result}")
        forward_e2e_samples.append((time.perf_counter() - start) * 1e3 / e2e_reps)
        np.testing.assert_allclose(forward_o, expected_o, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(forward_lse, expected_lse, rtol=4e-5, atol=4e-5)

    return {
        "schema": "tessera.nvidia.saved-lse-backward-benchmark.v1",
        "work_item": "NVIDIA-ATTENTION-LSE-BACKWARD-2026-10-01",
        "target": "nvidia_sm120",
        "architecture": backward.native_image.architecture,
        "device": subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
                                  "--format=csv,noheader"], check=True, capture_output=True,
                                 text=True).stdout.strip(),
        "shape_b_hq_hkv_sq_sk_d_dv": list(shape),
        "route": "GraphIR->ScheduleIR->TileIR->NVIDIA Target IR->PTX",
        "checkpoint": "saved forward output and row_lse from native forward package; deterministic_direct backward",
        "correctness": {"forward_execution_kind": fwd["execution_kind"],
                        "saved_backward_execution_kind": saved["execution_kind"],
                        "recompute_backward_execution_kind": recomputed["execution_kind"],
                        "forward_max_abs_error": float(np.max(np.abs(o - expected_o))),
                        "lse_max_abs_error": float(np.max(np.abs(lse - expected_lse))),
                        "gradient_max_abs_errors": errors,
                        "all_gradients_match_recompute_and_oracle": True},
        "packages": {"forward_abi": forward.launch_descriptor.abi_id,
                     "backward_abi": backward.launch_descriptor.abi_id,
                     "forward_image": forward.native_image.image_digest,
                     "backward_image": backward.native_image.image_digest,
                     "backward_schedule": backward.launch_descriptor.provenance.get("schedule_digest"),
                     "backward_tile": backward.launch_descriptor.provenance.get("tile_ir_digest"),
                     "recompute_abi": recompute.launch_descriptor.abi_id,
                     "recompute_image": recompute.native_image.image_digest,
                     "recompute_graph": recompute.launch_descriptor.provenance.get("graph_ir_digest"),
                     "recompute_schedule": recompute.launch_descriptor.provenance.get("schedule_digest"),
                     "recompute_tile": recompute.launch_descriptor.provenance.get("tile_ir_digest")},
        "timing": {"device_domain": "CUDA events around repeated native backward launches on caller-owned resident buffers and caller stream; transfers are outside the event window",
                   "device_samples_ms_per_launch": device_samples,
                   "device_median_ms": statistics.median(device_samples),
                   "device_cv": _cv(device_samples),
                   "end_to_end_domain": "runtime.launch on host arrays; includes binding, transfer, launch, and synchronization",
                   "end_to_end_samples_ms_per_launch": e2e_samples,
                   "end_to_end_median_ms": statistics.median(e2e_samples),
                   "end_to_end_cv": _cv(e2e_samples),
                   "samples": samples, "warmup": warmup,
                   "device_reps": device_reps, "end_to_end_reps": e2e_reps},
        "forward_timing": {
            "device_domain": "CUDA events around repeated native forward launches on resident buffers; transfers excluded",
            "device_samples_ms_per_launch": forward_device_samples,
            "device_median_ms": statistics.median(forward_device_samples),
            "end_to_end_domain": "runtime.launch host-array binding, transfers, launch and synchronization",
            "end_to_end_samples_ms_per_launch": forward_e2e_samples,
            "end_to_end_median_ms": statistics.median(forward_e2e_samples)},
        "recompute_timing": {
            "device_domain": "CUDA events around repeated native recompute launches on resident buffers; transfers excluded",
            "device_samples_ms_per_launch": recompute_device_samples,
            "device_median_ms": statistics.median(recompute_device_samples),
            "device_cv": _cv(recompute_device_samples),
            "end_to_end_domain": "runtime.launch host-array binding, transfers, launch and synchronization",
            "end_to_end_samples_ms_per_launch": recompute_e2e_samples,
            "end_to_end_median_ms": statistics.median(recompute_e2e_samples),
            "end_to_end_cv": _cv(recompute_e2e_samples)},
        "compiler_tool_hashes": {
            key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
            for key in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")
            if key in os.environ and Path(os.environ[key]).is_file()},
        "selector_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=20)
    parser.add_argument("--e2e-reps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = record(samples=args.samples, warmup=args.warmup,
                    device_reps=args.device_reps, e2e_reps=args.e2e_reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
