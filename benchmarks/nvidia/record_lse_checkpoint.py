"""Record the exact-SM120 saved-LSE versus recompute checkpoint packet.

The candidates are deliberately separate forward and backward native packages:
the saved lane owns f32[B,Hq,Sq] as an output/input pointer, while recompute
has no such buffer.  This recorder never promotes a selector; it records both
CUDA-event and end-to-end domains for the policy decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import _nvidia_native_descriptor_device_latency, _nvidia_native_descriptor_resources, launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_checkpoint_native import _backward_module, _forward_module


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _compile(module):
    return compile_graph_module(
        module, source_origin="NVIDIA-LSE-1", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )


def _sample(bundle, module, args: dict[str, object], *, samples: int, reps: int, warmup: int) -> dict[str, object]:
    descriptor, image = bundle.launch_descriptor, bundle.native_image
    assert descriptor is not None and image is not None
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    if not launch(artifact, args)["ok"]:
        raise RuntimeError(f"{descriptor.entry_symbol} smoke launch failed")
    device = [_nvidia_native_descriptor_device_latency(image, descriptor, args, reps=reps, warmup=warmup)
              for _ in range(samples)]
    e2e: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(reps):
            if not launch(artifact, args)["ok"]:
                raise RuntimeError(f"{descriptor.entry_symbol} end-to-end launch failed")
        e2e.append((time.perf_counter() - started) * 1e3 / reps)
    return {
        "entry": descriptor.entry_symbol,
        "abi_id": descriptor.abi_id,
        "device_event_samples_ms": device,
        "device_event_median_ms": _median(device),
        "end_to_end_samples_ms": e2e,
        "end_to_end_median_ms": _median(e2e),
        "resources": _nvidia_native_descriptor_resources(image, descriptor, block_size=128),
        "workspace_bytes": descriptor.workspace.bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not nvidia_cuda_host_ready():
        print(json.dumps({"status": "skipped", "reason": "host WSL CUDA SM120 unavailable"}))
        return 0
    if args.samples <= 0 or args.reps <= 0 or args.warmup < 0:
        raise ValueError("samples/reps must be positive and warmup nonnegative")
    modules = {
        "forward": {kind: _forward_module(saved=kind == "saved") for kind in ("recompute", "saved")},
        "backward": {kind: _backward_module(saved=kind == "saved") for kind in ("recompute", "saved")},
    }
    bundles = {stage: {kind: _compile(module) for kind, module in variants.items()}
               for stage, variants in modules.items()}
    rng = np.random.default_rng(120_101)
    q = np.ascontiguousarray((rng.normal(size=(1, 2, 3, 4)) * .2).astype(np.float32))
    k = np.ascontiguousarray((rng.normal(size=(1, 1, 4, 4)) * .2).astype(np.float32))
    v = np.ascontiguousarray((rng.normal(size=(1, 1, 4, 3)) * .2).astype(np.float32))
    do = np.ascontiguousarray((rng.normal(size=(1, 2, 3, 3)) * .2).astype(np.float32))
    scalars = {"B": 1, "Hq": 2, "Hkv": 1, "Sq": 3, "Sk": 4, "D": 4, "Dv": 3}
    row_lse = np.empty((1, 2, 3), dtype=np.float32)
    forward_args = {
        "saved": {"q": q, "k": k, "v": v, "o": np.empty((1, 2, 3, 3), np.float32), "row_lse": row_lse, **scalars},
        "recompute": {"q": q, "k": k, "v": v, "o": np.empty((1, 2, 3, 3), np.float32), **scalars},
    }
    # Populate the shared checkpoint exactly once before timing the consumer.
    saved_artifact = compile_result_from_bundle(bundles["forward"]["saved"], module=modules["forward"]["saved"]).to_runtime_artifact()
    if not launch(saved_artifact, forward_args["saved"])["ok"]:
        raise RuntimeError("saved-LSE producer setup failed")
    backward_args = {
        "saved": {"do": do, "q": q, "k": k, "v": v, "row_lse": row_lse,
                  "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v), **scalars},
        "recompute": {"do": do, "q": q, "k": k, "v": v,
                      "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v), **scalars},
    }
    rows = {
        stage: {kind: _sample(bundles[stage][kind], modules[stage][kind], stage_args,
                              samples=args.samples, reps=args.reps, warmup=args.warmup)
                for kind, stage_args in variants.items()}
        for stage, variants in (("forward", forward_args), ("backward", backward_args))
    }
    packet = {
        "schema": "tessera.nvidia.lse_checkpoint.benchmark.v1",
        "work_item": "NVIDIA-LSE-1",
        "device": "nvidia:sm_120",
        "shape": [1, 2, 1, 3, 4, 4, 3],
        "method": {"timing_domains": ["cuda_event", "end_to_end"], "samples": args.samples,
                   "repetitions": args.reps, "warmup": args.warmup,
                   "selector_default": "recompute", "ncu_required": True},
        "rows": rows,
        "selector_changed": False,
    }
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
