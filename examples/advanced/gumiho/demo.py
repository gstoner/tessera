from __future__ import annotations

import argparse

import numpy as np

from gumiho import (
    GumihoConfig,
    run_gumiho_demo,
    run_training_demo,
)
from gumiho.model import make_weights
from gumiho.resident import validate_resident_draft


def main() -> None:
    parser = argparse.ArgumentParser(description="Gumiho hybrid speculative decoding")
    parser.add_argument("--mode", default="decode",
                        choices=["step", "decode", "resident", "precision",
                                 "prefix", "forloop", "onchip"],
                        help="step: one validated speculative step; "
                             "decode: distill + multi-step decode with speedup; "
                             "resident: GPU-resident serial draft (one command "
                             "buffer per token); precision: f16/bf16 draft vs "
                             "f32; prefix: paged-KV prefix-sharing decode; "
                             "forloop: serial draft as one MPSGraph control-flow "
                             "(Phase-G Rung 1); onchip: composed on-device "
                             "speculative step (draft+verify MPSGraph + Rung-3 "
                             "MSL accept)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="apple_gpu",
                        choices=["apple_gpu", "apple_cpu", "numpy"],
                        help="compute backend for the draft + verify math")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16"],
                        help="half-precision dtype for --mode precision")
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

    if args.mode == "precision":
        from gumiho import run_precision_demo
        s = run_precision_demo(cfg, seed=args.seed, dtype=args.dtype)
        print("== Gumiho half-precision draft (f16/bf16 vs f32) ==")
        print(s)
        return

    if args.mode == "prefix":
        from gumiho import run_prefix_sharing_demo
        weights = make_weights(cfg, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        prompts = rng.integers(0, cfg.vocab, size=(args.prompts, cfg.context_len),
                               dtype=np.int64)
        s = run_prefix_sharing_demo(cfg, weights, prompts=prompts,
                                    max_new_tokens=args.max_new_tokens, seed=args.seed)
        print("== Gumiho paged-KV prefix-sharing decode ==")
        print(s)
        return

    if args.mode == "onchip":
        from gumiho import run_onchip_step_demo
        s = run_onchip_step_demo(cfg, seed=args.seed, target=args.target,
                                 distill_steps=300)
        print("== Gumiho on-device speculative step (composed kernels) ==")
        print(s)
        print("pipeline = serial_draft(forLoop, Rung 1) + parallel + tree_verify "
              "(MPSGraph) -> spec_accept (MSL, Rung 3)")
        return

    if args.mode == "forloop":
        from gumiho import validate_serial_forloop
        weights = make_weights(cfg, seed=args.seed)
        r = validate_serial_forloop(cfg, weights, seed=args.seed)
        print("== Gumiho serial draft as one MPSGraph control-flow (Phase-G Rung 1) ==")
        print(f"backend={r.backend} tokens={r.tokens} matches_host={r.matches_host} "
              f"(max_hidden_err={r.max_hidden_err:.2e})")
        print(f"dispatches={r.dispatches} (one forLoop graph) vs "
              f"~{r.host_dispatch_equiv} per-op host dispatches — the whole "
              f"autoregressive serial loop is lowered into a single kernel")
        return

    if args.mode == "resident":
        weights = make_weights(cfg, seed=args.seed)
        r = validate_resident_draft(cfg, weights, seed=args.seed)
        print("== Gumiho GPU-resident serial draft (one command buffer/token) ==")
        print(f"backend={r.backend} tokens={r.tokens} matches_host={r.matches_host} "
              f"(max_logit_err={r.max_logit_abs_err:.2e})")
        reduction = r.host_dispatch_equiv / max(r.command_buffers, 1)
        print(f"command_buffers={r.command_buffers} vs "
              f"~{r.host_dispatch_equiv} per-op host dispatches "
              f"({reduction:.0f}x fewer GPU syncs); weights resident, only the "
              f"token id + carry hidden read back")
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
