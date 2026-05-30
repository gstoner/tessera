from __future__ import annotations

import argparse

from gumiho import GumihoConfig, run_gumiho_demo, run_training_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gumiho hybrid speculative decoding")
    parser.add_argument("--mode", default="decode", choices=["step", "decode"],
                        help="step: one validated speculative step; "
                             "decode: distill + multi-step decode with speedup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="apple_gpu",
                        choices=["apple_gpu", "apple_cpu", "numpy"],
                        help="compute backend for the draft + verify math")
    parser.add_argument("--serial-tokens", type=int, default=2)
    parser.add_argument("--parallel-heads", type=int, default=5)
    parser.add_argument("--top-paths", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    cfg = GumihoConfig(
        serial_tokens=args.serial_tokens,
        parallel_heads=args.parallel_heads,
        fta_top_paths=args.top_paths,
    )

    if args.mode == "step":
        summary = run_gumiho_demo(cfg, seed=args.seed, target=args.target)
        print("== Gumiho hybrid speculative decoding — one validated step ==")
        print(summary)
        print(f"schedule = serial_head({cfg.serial_tokens}) + "
              f"parallel_heads({cfg.parallel_heads}) -> FTA top-{cfg.fta_top_paths} "
              f"-> tree_verify -> accept -> advance_kv")
        return

    before, after = run_training_demo(
        cfg, seed=args.seed, target=args.target,
        num_prompts=args.prompts, max_new_tokens=args.max_new_tokens)
    print("== Gumiho multi-step speculative decode (distill -> measure) ==")
    print(before)
    print(after)
    gain = after.speedup_vs_vanilla / max(before.speedup_vs_vanilla, 1e-9)
    print(f"distillation lifted tokens/target-pass {before.tokens_per_step:.2f} "
          f"-> {after.tokens_per_step:.2f} ({gain:.2f}x), vanilla autoregressive = 1.00")


if __name__ == "__main__":
    main()
