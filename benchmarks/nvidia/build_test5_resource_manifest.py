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
    "nvidia_generic_cuda": ("tessera_nvidia_fused_kernel",),
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


def existing_routes(manifest: dict[str, Any], routes: list[str]) -> list[str]:
    """The ``routes`` already present in ``manifest`` (routes, details or a
    source tagged with the route) -- what a capture must refuse up front
    unless it was asked to refresh them."""
    tagged = {s.get("route") for s in manifest.get("sources", [])}
    return sorted(r for r in routes
                  if r in manifest.get("routes", {}) or r in manifest.get("details", {})
                  or r in tagged)


def add_isolated_routes(manifest: dict[str, Any],
                        isolated: dict[str, dict[str, Any]], *,
                        refresh: bool = False) -> dict[str, Any]:
    """Add routes each captured in a report of its OWN (``route -> payload``).

    Every kernel in such a report belongs to that route
    (``profile_route_resources.py`` brackets one route's launches with
    ``cuProfilerStart/Stop``), so no name pattern is consulted: kernels that
    share a name across storages -- the tf32 and fp8 builds of one emitted
    lane -- stay apart. An isolated report with no kernel is refused (a route
    that launched nothing has no resource evidence).

    A route already in the manifest is refused unless ``refresh``. With
    ``refresh`` exactly the routes in ``isolated`` are replaced -- their
    ``routes`` and ``details`` entries and every ``sources`` entry tagged with
    them are dropped before the new capture is added, so one route never
    mixes an old and a new capture -- and every other route, and every
    untagged source, is kept byte-for-byte. Refreshing a route that is not
    present is refused too: a refresh names what it replaces.
    """
    present = existing_routes(manifest, list(isolated))
    if present and not refresh:
        raise ValueError(
            f"{', '.join(present)} already in the manifest; refusing to "
            "overwrite (pass --refresh to replace exactly these routes)")
    if refresh:
        missing = sorted(set(isolated) - set(present))
        if missing:
            raise ValueError(f"--refresh names routes not in the manifest: {', '.join(missing)}")
    replaced = set(isolated) if refresh else set()
    out = {**manifest,
           "sources": [s for s in manifest.get("sources", [])
                       if s.get("route") not in replaced],
           "routes": {k: v for k, v in manifest.get("routes", {}).items()
                      if k not in replaced},
           "details": {k: v for k, v in manifest.get("details", {}).items()
                       if k not in replaced}}
    for route, payload in sorted(isolated.items()):
        rows = list(payload.get("rows", []))
        if not rows:
            raise ValueError(f"isolated report for {route} holds no kernel")
        out["details"][route] = rows
        out["routes"][route] = [row["resource_fingerprint"] for row in rows]
        if payload.get("source"):
            out["sources"].append({"name": payload.get("source"),
                                   "sha256": payload.get("source_sha256"),
                                   "route": route})
    return out


def _route_arg(text: str) -> tuple[str, Path]:
    route, sep, path = text.partition("=")
    if not sep or not route or not path:
        raise argparse.ArgumentTypeError("expected ROUTE=payload.json")
    return route, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*",
                        help="normalized payloads mapped to routes by kernel name")
    parser.add_argument("--base", type=Path,
                        help="an existing manifest to extend instead of building one")
    parser.add_argument("--route", type=_route_arg, action="append", default=[],
                        help="ROUTE=payload.json: a report holding only that route")
    parser.add_argument("--refresh", action="store_true",
                        help="replace exactly the --route routes already in --base")
    parser.add_argument("--check-routes", nargs="+", metavar="ROUTE",
                        help="preflight only: exit 1 (naming them) if any ROUTE is "
                             "already in --base and --refresh was not given, or if "
                             "--refresh names a route that is not there")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.base and args.inputs:
        parser.error("--base extends an existing manifest; do not also pass inputs")
    if args.check_routes:
        if not args.base:
            parser.error("--check-routes needs --base")
        base = json.loads(args.base.read_text())
        present = existing_routes(base, args.check_routes)
        if present and not args.refresh:
            print(f"already in {args.base}: {', '.join(present)}; pass --refresh "
                  "to replace exactly these routes")
            return 1
        missing = sorted(set(args.check_routes) - set(present))
        if args.refresh and missing:
            print(f"--refresh names routes not in {args.base}: {', '.join(missing)}")
            return 1
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.base:
        manifest = json.loads(args.base.read_text())
    else:
        manifest = build([json.loads(path.read_text()) for path in args.inputs])
    if args.route:
        manifest = add_isolated_routes(manifest, {
            route: json.loads(path.read_text()) for route, path in args.route},
            refresh=args.refresh)
    elif args.refresh:
        parser.error("--refresh needs --route")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
