"""Runtime smoke command for Tessera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tessera.runtime import runtime_smoke_telemetry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessera-runtime-smoke",
        description="Exercise the CPU runtime spine and emit telemetry JSON.",
    )
    parser.add_argument("--native", action="store_true", help="Use native runtime library if found")
    parser.add_argument("--bytes", type=int, default=64, help="Buffer size for memory smoke")
    parser.add_argument("--output", help="Write telemetry JSON to this path")
    args = parser.parse_args(argv)

    payload = runtime_smoke_telemetry(mock=not args.native, bytes_size=args.bytes)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
