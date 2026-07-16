"""Map normalized Nsight kernels to TEST-5 production candidate routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_ROUTES = {
    "nvidia_tile_matmul_direct": ("tessera_tile_matmul_direct",),
    "nvidia_tile_matmul_shared": ("tessera_tile_matmul_shared",),
    "nvidia_mma_gemm_shipped": ("^gemm",),
    "nvidia_mma_gemm_emitted": ("mma_gemm",),
    "nvidia_mma_fused": ("tessera_nvidia_mma_fused_kernel",),
    "nvidia_mma_attn": ("tessera_nvidia_mma_attn_kernel",),
    "nvidia_mma_fused_composed_tf32": ("gemm", "epi("),
    "nvidia_mma_attn_composed_tf32": ("gemm", "scale_mask", "softmax"),
    "nvidia_mma_gated_composed_tf32": ("gemm", "gate("),
    "direct": ("conv_direct",),
    "generated_atomic_vjp": ("tsr_flash_bwd",),
    "generated_row_reduce": ("tsr_reduce_kernel",),
    "generated_gather": ("gather_k",),
    "generated_combine": ("combine_k",),
    "generated_grouped": ("gg_k",),
    "fused_paged_attention": ("paged_attn",),
    "staged_paged_attention": ("mm_f32", "scale_mask", "softmax"),
    "async_ring": ("out(Q",),
}


def build(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    kernels: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("rows", []):
            kernels[row["kernel"]] = row
    routes: dict[str, list[str]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for route, patterns in _ROUTES.items():
        matched = [row for name, row in kernels.items()
                   if any((name == pattern[1:] if pattern.startswith("^")
                           else pattern in name) for pattern in patterns)]
        if matched:
            details[route] = matched
            routes[route] = [row["resource_fingerprint"] for row in matched]
    return {"schema": "tessera.nvidia.route-resources.v1",
            "sources": [{"name": payload.get("source"),
                         "sha256": payload.get("source_sha256")}
                        for payload in payloads if payload.get("source")],
            "routes": routes, "details": details}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    args.output.write_text(json.dumps(build(payloads), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
