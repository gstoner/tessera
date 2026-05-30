from __future__ import annotations

import argparse

from kv_cache_serving.compression import CachePolicy, estimate_request_cache
from kv_cache_serving.scheduler import Request, route_requests
from kv_cache_serving.serving import run_serving_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefixes", type=int, default=6)
    parser.add_argument(
        "--plan-only", action="store_true",
        help="print the routing/accounting plan without running the real cache",
    )
    args = parser.parse_args()

    policy = CachePolicy(k_bits=4, v_bits=4, residual_bits=1, retrieval_head_fraction=0.25, streaming_window=4096)
    requests = [
        Request(request_id=f"req_{i:03d}", tenant=f"tenant_{i % 4}", prefix_id=f"prefix_{i % args.prefixes}", context_tokens=args.context)
        for i in range(args.requests)
    ]
    placements = route_requests(requests, decode_workers=4)
    cache = estimate_request_cache(args.heads, args.head_dim, args.context, policy)

    print("== routing + accounting plan ==")
    print(f"policy={policy}")
    print(f"cache_per_request_mib={cache / 2**20:.2f}")
    for placement in placements[:12]:
        print(placement)

    if args.plan_only:
        return

    # Execute the plan against the real block-paged cache manager.
    print("\n== real block-paged serving (tessera.cache.MLABlockPagedCache) ==")
    summary = run_serving_demo(
        num_requests=args.requests,
        num_prefixes=args.prefixes,
        policy=policy,
        accounting_heads=args.heads,
        accounting_head_dim=args.head_dim,
        accounting_context=args.context,
    )
    print(summary)
    print(f"prefix sharing reclaimed {summary.blocks_saved} of "
          f"{summary.blocks_without_sharing} would-be blocks")


if __name__ == "__main__":
    main()
