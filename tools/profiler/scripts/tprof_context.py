#!/usr/bin/env python3
"""Sample Tessera profiler context and emit tessera.profiler_context.v1 JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_collectors import collect_profiler_context

    parser = argparse.ArgumentParser(
        prog="tprof-context",
        description="Emit a tessera.profiler_context.v1 artifact from mock/file/native collectors.",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "nvidia", "rocm", "apple"),
        default="mock",
        help="Context provider to sample.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target name for mock/file samples; native providers infer their target.",
    )
    parser.add_argument(
        "--input",
        help="Load a full context artifact or raw sample/list JSON instead of native sampling.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Native GPU index for nvidia/rocm providers.",
    )
    parser.add_argument("--out", help="Write JSON artifact to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        artifact = collect_profiler_context(
            args.provider,
            target=args.target,
            input_path=args.input,
            device_index=args.device_index,
        )
    except Exception as exc:
        parser.error(str(exc))
    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
