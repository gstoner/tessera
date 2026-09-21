from __future__ import annotations

import argparse

from speculative_decoding.tree import DraftModel, TargetModel, speculative_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The next research direction is")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branching", type=int, default=3)
    args = parser.parse_args()

    draft = DraftModel()
    target = TargetModel()
    accepted = speculative_step(args.prompt, draft, target, depth=args.depth, branching=args.branching)
    print(f"prompt={args.prompt!r}")
    print(f"accepted_tokens={accepted}")
    print("schedule=expand_tree -> batch_verify -> compact_accepts -> advance_kv")


if __name__ == "__main__":
    main()
