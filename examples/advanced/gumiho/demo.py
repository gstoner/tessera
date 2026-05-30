from __future__ import annotations

import argparse

from gumiho import GumihoConfig, run_gumiho_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gumiho hybrid speculative decoding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="apple_gpu",
                        choices=["apple_gpu", "apple_cpu", "numpy"],
                        help="compute backend for the draft + verify math")
    parser.add_argument("--serial-tokens", type=int, default=2)
    parser.add_argument("--parallel-heads", type=int, default=5)
    parser.add_argument("--top-paths", type=int, default=8)
    args = parser.parse_args()

    cfg = GumihoConfig(
        serial_tokens=args.serial_tokens,
        parallel_heads=args.parallel_heads,
        fta_top_paths=args.top_paths,
    )
    summary = run_gumiho_demo(cfg, seed=args.seed, target=args.target)

    print("== Gumiho hybrid speculative decoding (Apple backend) ==")
    print(summary)
    print(f"schedule = serial_head({cfg.serial_tokens}) + "
          f"parallel_heads({cfg.parallel_heads}) -> FTA top-{cfg.fta_top_paths} "
          f"-> tree_verify -> accept -> advance_kv")


if __name__ == "__main__":
    main()
