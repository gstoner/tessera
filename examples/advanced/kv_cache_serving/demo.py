from __future__ import annotations

import argparse

from kv_cache_serving.compression import CachePolicy, estimate_request_cache
from kv_cache_serving.scheduler import Request, route_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    policy = CachePolicy(k_bits=4, v_bits=4, residual_bits=1, retrieval_head_fraction=0.25, streaming_window=4096)
    requests = [
        Request(request_id=f"req_{i:03d}", tenant=f"tenant_{i % 4}", prefix_id=f"prefix_{i % 6}", context_tokens=args.context)
        for i in range(args.requests)
    ]
    placements = route_requests(requests, decode_workers=4)
    cache = estimate_request_cache(args.heads, args.head_dim, args.context, policy)

    print(f"policy={policy}")
    print(f"cache_per_request_mib={cache / 2**20:.2f}")
    for placement in placements[:12]:
        print(placement)


if __name__ == "__main__":
    main()
