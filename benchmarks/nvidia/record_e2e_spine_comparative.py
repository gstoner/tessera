"""Record stable canonical-versus-production softmax/attention/MoE evidence.

The first lifecycle launch is discarded. Retained device-event samples amortize
1,000 resident launches and end-to-end samples amortize ten allocation/copy
inclusive launches. This recorder is evidence-only: Tessera has no registered
material-benefit threshold for these E2E foundation routes, so stability and
cross-domain winner consensus cannot independently promote a selector.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_comparative.json"
POLICY = 0.04


def _delta(a: float, b: float) -> float:
    return abs(a - b) / min(a, b) if min(a, b) else 0.0


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _softmax_module(shape: tuple[int, int]):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
    rows, columns = shape
    x = IRType(f"tensor<{rows}x{columns}xf32>", (str(rows), str(columns)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="comparative_softmax", args=[IRArg("x", x)], result_types=[x],
        body=[IROp(result="o", op_name="tessera.softmax", operands=["%x"],
                   operand_types=[str(x)], result_type=str(x), kwargs={"axis": -1})],
        return_values=["%o"],
    )])


def _attention_module(shape: tuple[int, ...]):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
    b, hq, hkv, sq, sk, d, dv = shape
    q = IRType(f"tensor<{b}x{hq}x{sq}x{d}xf32>", tuple(map(str, (b, hq, sq, d))), "fp32")
    k = IRType(f"tensor<{b}x{hkv}x{sk}x{d}xf32>", tuple(map(str, (b, hkv, sk, d))), "fp32")
    v = IRType(f"tensor<{b}x{hkv}x{sk}x{dv}xf32>", tuple(map(str, (b, hkv, sk, dv))), "fp32")
    o = IRType(f"tensor<{b}x{hq}x{sq}x{dv}xf32>", tuple(map(str, (b, hq, sq, dv))), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="comparative_attention", args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=[o], body=[IROp(
            result="o", op_name="tessera.flash_attn", operands=["%q", "%k", "%v"],
            operand_types=[str(q), str(k), str(v)], result_type=str(o),
            kwargs={"scale": d ** -0.5, "causal": True},
        )], return_values=["%o"],
    )])


def _production_resources(source: str, entry: str, threads: int, schedule: str) -> dict[str, Any]:
    from benchmarks.nvidia.record_tile_fragment_resources import _CudaOccupancy, _artifact_row
    with tempfile.TemporaryDirectory(prefix="tessera-e2e-compare-") as tmp:
        cu = Path(tmp) / "candidate.cu"; cubin = Path(tmp) / "candidate.cubin"
        cu.write_text(source)
        subprocess.run(["/usr/local/cuda/bin/nvcc", "-arch=sm_120", "-O3", "-cubin",
                        str(cu), "-o", str(cubin)], check=True)
        occupancy = _CudaOccupancy()
        try:
            return _artifact_row(cubin, entry, threads, occupancy, schedule=schedule)
        finally:
            occupancy.close()


def _measure(candidates: dict[str, dict[str, Any]], *, samples: int,
             device_reps: int, e2e_reps: int, warmup: int) -> tuple[dict, dict]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120
    names = tuple(candidates)
    raw: dict[str, list[dict[str, list[float]]]] = {
        name: [{"device": [], "e2e": []}, {"device": [], "e2e": []}]
        for name in names
    }
    for sample in range(samples):
        for run in ((0, 1) if sample % 2 == 0 else (1, 0)):
            order = names if (sample + run) % 2 == 0 else names[::-1]
            for name in order:
                condition_sm120(reps=20)
                retained_reps = device_reps * int(
                    candidates[name].get("device_reps_multiplier", 1))
                raw[name][run]["device"].append(
                    float(candidates[name]["device"](retained_reps, warmup)))
                candidates[name]["e2e"]()  # discard first lifecycle launch
                started = time.perf_counter()
                for _ in range(e2e_reps):
                    candidates[name]["e2e"]()
                raw[name][run]["e2e"].append(
                    (time.perf_counter() - started) * 1e3 / e2e_reps)
    runs: dict[str, list[dict[str, Any]]] = {name: [{
        "run": run + 1,
        "device_event_ms": statistics.median(raw[name][run]["device"]),
        "end_to_end_ms": statistics.median(raw[name][run]["e2e"]),
        "device_samples_ms": raw[name][run]["device"],
        "end_to_end_samples_ms": raw[name][run]["e2e"],
    } for run in range(2)] for name in names}
    winners = {domain: [min(names, key=lambda name: float(runs[name][run][domain]))
                        for run in range(2)]
               for domain in ("device_event_ms", "end_to_end_ms")}
    return runs, winners


def record(*, samples: int = 15, device_reps: int = 1000,
           e2e_reps: int = 10, warmup: int = 100) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.emit import nvidia_cuda as nv

    bridge = rt._load_nvidia_ptx_launch()
    if bridge is None:
        raise RuntimeError("SM120 PTX bridge is unavailable")
    rows: list[dict[str, Any]] = []

    def add_case(op: str, shape: tuple[int, ...], module, arrays: dict[str, np.ndarray],
                 scalars: dict[str, int], production: dict[str, Callable],
                 production_resource: dict[str, Any]) -> None:
        cold_started = time.perf_counter()
        bundle = compile_graph_module(module, source_origin="NVIDIA-E2E-2",
            target="nvidia_sm120", options={"package_native": True},
            enable_tool_validation=False)
        cold_ms = (time.perf_counter() - cold_started) * 1e3
        warm_started = time.perf_counter()
        warm = compile_graph_module(module, source_origin="NVIDIA-E2E-2",
            target="nvidia_sm120", options={"package_native": True},
            enable_tool_validation=False)
        warm_ms = (time.perf_counter() - warm_started) * 1e3
        assert bundle.native_image and bundle.launch_descriptor and warm.native_image
        artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
        bindings = {**arrays, **scalars}
        result = rt.launch(artifact, bindings)
        if not result["ok"]:
            raise RuntimeError(str(result.get("reason")))
        ordered = sorted(bundle.launch_descriptor.buffers, key=lambda item: item.ordinal)
        cbuf = (ctypes.c_void_p * len(ordered))(
            *(int(arrays[item.name].ctypes.data) for item in ordered))
        dim_names = [item.name for item in sorted(bundle.launch_descriptor.scalars,
                                                  key=lambda item: item.ordinal)]
        dims = (ctypes.c_int64 * len(dim_names))(*(scalars[name] for name in dim_names))
        entry = bundle.launch_descriptor.entry_symbol.encode()
        def canonical_device(reps: int, warmup_count: int) -> float:
            latency = ctypes.c_float()
            rc = bridge.tessera_nvidia_ptx_benchmark(
                entry, cbuf, len(ordered), dims, len(dim_names), warmup_count,
                reps, ctypes.byref(latency))
            if rc:
                raise RuntimeError(f"canonical {op} benchmark rc={rc}")
            return float(latency.value)
        def canonical_e2e() -> None:
            launched = rt.launch(artifact, bindings)
            if not launched["ok"]:
                raise RuntimeError(str(launched.get("reason")))
        candidates: dict[str, dict[str, Any]] = {
            "canonical_tile": {"device": canonical_device, "e2e": canonical_e2e},
            "production_cuda": production,
        }
        measured, winners = _measure(candidates, samples=samples,
                                     device_reps=device_reps,
                                     e2e_reps=e2e_reps, warmup=warmup)
        resources = {
            "canonical_tile": bundle.native_image.resource_record.to_dict()
            if bundle.native_image.resource_record else None,
            "production_cuda": production_resource,
        }
        for candidate in candidates:
            candidate_runs = measured[candidate]
            stability = {domain: _delta(candidate_runs[0][domain], candidate_runs[1][domain])
                         for domain in ("device_event_ms", "end_to_end_ms")}
            consensus = all(winners[domain] == [candidate, candidate]
                            for domain in winners)
            rows.append({
                "op": op, "shape": list(shape), "candidate": candidate,
                "runs": candidate_runs, "stability_fraction": stability,
                "stable": all(value <= POLICY for value in stability.values()),
                "winner_consensus": consensus,
                "selector_eligible": False,
                "selector_disposition": "retain_existing_no_registered_materiality_threshold",
                "selector_changed": False, "resources": resources[candidate],
                "resource_fingerprint": _fingerprint(resources[candidate]),
                "image_digest": bundle.native_image.image_digest
                if candidate == "canonical_tile" else None,
                "compile": {"cold_ms": cold_ms, "warm_ms": warm_ms,
                            "cold_state": bundle.native_image.compile_state,
                            "warm_state": warm.native_image.compile_state},
                "device_repetitions_per_sample": device_reps * int(
                    candidates[candidate].get("device_reps_multiplier", 1)),
            })

    softmax_resource = _production_resources(
        nv._synthesize_softmax_cuda(), "tsr_softmax_kernel", 256,
        "production_cooperative_256")
    for softmax_shape in ((256, 1024), (127, 259)):
        rng = np.random.default_rng(sum(softmax_shape)); x = rng.standard_normal(softmax_shape, dtype=np.float32)
        o = np.empty_like(x)
        add_case("softmax", softmax_shape, _softmax_module(softmax_shape), {"x": x, "o": o},
                 {"Rows": softmax_shape[0], "K": softmax_shape[1]}, {
                     "device": lambda reps, warmup, x=x: nv.measure_row_softmax_device(
                         x, reps=reps, warmup=warmup),
                     "e2e": lambda x=x: nv.run_row_softmax(x),
                 }, softmax_resource)

    attention_resource = _production_resources(
        nv._synthesize_flash_fwd_multiwarp_cuda(4),
        "tessera_nvidia_flash_attn_fwd_w4_kernel", 128,
        "production_warp_per_query_w4")
    for attention_shape in ((1, 4, 2, 32, 64, 64, 64),
                            (1, 4, 1, 17, 31, 32, 29)):
        b,hq,hkv,sq,sk,d,dv = attention_shape; rng=np.random.default_rng(sum(attention_shape))
        q=rng.standard_normal((b,hq,sq,d),dtype=np.float32)*.1
        k=rng.standard_normal((b,hkv,sk,d),dtype=np.float32)*.1
        v=rng.standard_normal((b,hkv,sk,dv),dtype=np.float32)*.1
        o=np.empty((b,hq,sq,dv),np.float32); scale=d**-.5
        add_case("attention_forward", attention_shape, _attention_module(attention_shape),
                 {"q":q,"k":k,"v":v,"o":o},
                 {"B":b,"Hq":hq,"Hkv":hkv,"Sq":sq,"Sk":sk,"D":d,"Dv":dv}, {
                     "device": lambda reps,warmup,q=q,k=k,v=v,scale=scale:
                         nv.measure_flash_attention_forward_schedule_device(
                             q,k,v,scale=scale,causal=True,warps_per_cta=4,
                             reps=reps,warmup=warmup),
                     "e2e": lambda q=q,k=k,v=v,scale=scale:
                         nv.run_flash_attention_forward_schedule(
                             q,k,v,scale=scale,causal=True,warps_per_cta=4),
                 }, attention_resource)

    from tessera.compiler.nvidia_native import package_moe_kernels

    moe_source = nv._synthesize_moe_cuda()
    moe_resources = {
        "dispatch": _production_resources(
            moe_source, "gather_k", 256, "production_generated_gather"),
        "combine": _production_resources(
            moe_source, "combine_k", 256, "production_generated_combine"),
        "grouped_gemm": _production_resources(
            moe_source, "gg_k", 256, "production_generated_grouped"),
    }
    tokens, slots, hidden, experts, kdim, ndim = 257, 389, 193, 5, 193, 127
    offsets = np.array([0, 79, 79, 180, 281, 389], dtype=np.int32)
    cold_started = time.perf_counter()
    moe_packages = package_moe_kernels(
        num_tokens=tokens, num_slots=slots, hidden=hidden,
        expert_count=experts, expert_k=kdim, expert_n=ndim,
        group_offsets=tuple(int(value) for value in offsets),
        pipeline_name="tessera-nvidia-pipeline-sm120", storage="fp32",
    )
    cold_ms = (time.perf_counter() - cold_started) * 1e3
    warm_started = time.perf_counter()
    warm_packages = package_moe_kernels(
        num_tokens=tokens, num_slots=slots, hidden=hidden,
        expert_count=experts, expert_k=kdim, expert_n=ndim,
        group_offsets=tuple(int(value) for value in offsets),
        pipeline_name="tessera-nvidia-pipeline-sm120", storage="fp32",
    )
    warm_ms = (time.perf_counter() - warm_started) * 1e3
    rng = np.random.default_rng(8471)
    x = (rng.standard_normal((tokens, hidden)) * .2).astype(np.float32)
    token_of_slot = rng.integers(0, tokens, size=slots, dtype=np.int32)
    dispatched = np.empty((slots, hidden), dtype=np.float32)
    weights = rng.random(slots, dtype=np.float32)
    combined = np.empty((tokens, hidden), dtype=np.float32)
    grouped_x = (rng.standard_normal((slots, kdim)) * .2).astype(np.float32)
    expert_weights = (rng.standard_normal((experts, kdim, ndim)) * .1).astype(np.float32)
    grouped_o = np.empty((slots, ndim), dtype=np.float32)
    moe_cases = (
        ("moe_dispatch", "dispatch", moe_packages[0], warm_packages[0],
         {"X": x, "token_of_slot": token_of_slot, "dispatched": dispatched},
         {"Tokens": tokens, "Slots": slots, "Hidden": hidden},
         {"device": lambda reps, warmup: nv.measure_moe_dispatch_device(
              x, token_of_slot, reps=reps),
          "e2e": lambda: nv.run_moe_dispatch_f32(x, token_of_slot)},
         (tokens, slots, hidden)),
        ("moe_combine", "combine", moe_packages[1], warm_packages[1],
         {"partials": dispatched, "token_of_slot": token_of_slot,
          "combine_weights": weights, "O": combined},
         {"Tokens": tokens, "Slots": slots, "Hidden": hidden},
         {"device": lambda reps, warmup: nv.measure_moe_combine_device(
              dispatched, token_of_slot, weights, tokens, reps=reps),
          "e2e": lambda: nv.run_moe_combine_f32(
              dispatched, token_of_slot, weights, tokens)},
         (tokens, slots, hidden)),
        ("moe_grouped_gemm", "grouped_gemm", moe_packages[2], warm_packages[2],
         {"X": grouped_x, "W": expert_weights, "group_offsets": offsets,
          "O": grouped_o},
         {"GroupedTokens": slots, "K": kdim, "N": ndim, "Experts": experts},
         {"device": lambda reps, warmup: nv.measure_grouped_gemm_device(
              grouped_x, expert_weights, np.diff(offsets), reps=reps),
          "e2e": lambda: nv.run_grouped_gemm_f32(
              grouped_x, expert_weights, np.diff(offsets))},
         (slots, kdim, ndim, experts)),
    )
    for op, route, package, warm_package, arrays, scalars, production, case_shape in moe_cases:
        bindings = {**arrays, **scalars}
        canonical_e2e = (
            lambda package=package, arrays=arrays, scalars=scalars:
            rt._submit_nvidia_sm120_native(
                package.image, package.descriptor, arrays, scalars, None)
        )
        candidates: dict[str, dict[str, Any]] = {
            "canonical_tile": {
                "device": lambda reps, warmup, package=package, bindings=bindings:
                    rt._nvidia_native_descriptor_device_latency(
                        package.image, package.descriptor, bindings,
                        reps=reps, warmup=warmup),
                "e2e": canonical_e2e,
            },
            "production_cuda": production,
        }
        if op == "moe_combine":
            for value in candidates.values():
                value["device_reps_multiplier"] = 10
        measured, winners = _measure(
            candidates, samples=samples, device_reps=device_reps,
            e2e_reps=e2e_reps, warmup=warmup)
        resources = {
            "canonical_tile": package.image.resource_record.to_dict()
            if package.image.resource_record else None,
            "production_cuda": moe_resources[route],
        }
        for candidate in candidates:
            candidate_runs = measured[candidate]
            stability = {
                domain: _delta(candidate_runs[0][domain], candidate_runs[1][domain])
                for domain in ("device_event_ms", "end_to_end_ms")
            }
            consensus = all(
                winners[domain] == [candidate, candidate] for domain in winners)
            rows.append({
                "op": op, "shape": list(case_shape), "candidate": candidate,
                "runs": candidate_runs,
                "stability_fraction": stability,
                "stable": all(value <= POLICY for value in stability.values()),
                "winner_consensus": consensus, "selector_eligible": False,
                "selector_disposition": "retain_existing_no_registered_materiality_threshold",
                "selector_changed": False, "resources": resources[candidate],
                "resource_fingerprint": _fingerprint(resources[candidate]),
                "image_digest": package.image.image_digest
                if candidate == "canonical_tile" else None,
                "compile": {"cold_ms": cold_ms, "warm_ms": warm_ms,
                            "cold_state": package.image.compile_state,
                            "warm_state": warm_package.image.compile_state},
                "device_repetitions_per_sample": device_reps * int(
                    candidates[candidate].get("device_reps_multiplier", 1)),
            })
    device = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    return {"schema":"tessera.nvidia.e2e-spine-comparative.v1",
            "work_item":"NVIDIA-E2E-2", "device":device,
            "stability_policy":{"relative_fraction":POLICY,"runs":2,
                "required_domains":["device_event_ms","end_to_end_ms"]},
            "promotion_policy":{"cross_domain_consensus_required":True,
                "material_benefit_threshold":"unregistered",
                "effect":"no candidate is selector-eligible"},
            "method":{"samples_per_run":samples,"device_repetitions_per_sample":device_reps,
                "end_to_end_repetitions_per_sample":e2e_reps,"warmup":warmup,
                "discarded_end_to_end_launches_per_sample":1,
                "sampling":"per-candidate-clock-conditioned_rotated_disjoint_cohorts"},
            "selector_promotions":[],"rows":rows}


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples",type=int,default=15);p.add_argument("--device-reps",type=int,default=1000)
    p.add_argument("--e2e-reps",type=int,default=10);p.add_argument("--warmup",type=int,default=100)
    p.add_argument("--output",type=Path,default=OUT);a=p.parse_args(argv)
    payload=record(samples=a.samples,device_reps=a.device_reps,e2e_reps=a.e2e_reps,warmup=a.warmup)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(f"wrote {a.output} ({len(payload['rows'])} rows)");return 0


if __name__ == "__main__":
    raise SystemExit(main())
