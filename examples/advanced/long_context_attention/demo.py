from __future__ import annotations

import argparse

from long_context_attention.planner import classify_heads, estimate_cache_bytes, synthetic_head_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--sink", type=int, default=128)
    args = parser.parse_args()

    stats = synthetic_head_stats(args.heads)
    roles = classify_heads(stats)
    full = estimate_cache_bytes(args.heads, args.seq_len, args.head_dim, args.seq_len, 0)
    specialized = sum(
        estimate_cache_bytes(1, args.seq_len, args.head_dim, args.seq_len if role == "retrieval" else args.window, args.sink)
        for role in roles
    )

    for idx, role in enumerate(roles):
        print(f"head={idx:02d} role={role} stats={stats[idx]}")
    print(f"full_cache_gib={full / 2**30:.2f}")
    print(f"specialized_cache_gib={specialized / 2**30:.2f}")
    print(f"reduction={full / specialized:.2f}x")


if __name__ == "__main__":
    main()
