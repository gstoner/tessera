#!/usr/bin/env python3
"""P1 (2026-06-09) — Generate `apple_runtime_ops.inc` from
`tessera.compiler.apple_gpu_envelope._APPLE_GPU_RUNTIME_OPS`.

The C++ Tile→Apple GPU pass (`TileToApple.cpp::isAppleGpuRuntimeOp`)
used to carry a hand-maintained mirror of the Python runtime envelope.
This generator makes the envelope module the single source: the C++
list is an X-macro include regenerated whenever the envelope changes.

Sync gates:
  * ``tests/unit/test_apple_runtime_ops_table_in_sync.py`` — Python-only
    (no build needed), fails when the .inc disagrees with the envelope.
  * ``tests/unit/test_apple_gpu_tile_pass_status_matches_envelope.py`` —
    runs the real ``tessera-opt`` pass over every envelope op.

Run:
    python scripts/generate_apple_runtime_ops_table.py
    python scripts/generate_apple_runtime_ops_table.py --check
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler.apple_gpu_envelope import (  # noqa: E402
    _APPLE_GPU_RUNTIME_OPS,
    APPLE_GPU_LANE_BY_OP,
)

OUT = (
    ROOT / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend" / "include"
    / "Tessera" / "Target" / "Apple" / "apple_runtime_ops.inc"
)

HEADER = """\
//===- apple_runtime_ops.inc - Apple GPU runtime envelope ------*- C++ -*-===//
//
// Auto-generated from `python/tessera/compiler/apple_gpu_envelope.py`
// (_APPLE_GPU_RUNTIME_OPS).  Do NOT edit by hand — regenerate with:
//
//     python scripts/generate_apple_runtime_ops_table.py
//
// X-macro format: each line is a TESSERA_APPLE_GPU_RUNTIME_OP call the
// includer expands.  Grouped by runtime dispatch lane for readability.
//
// Sync gates: tests/unit/test_apple_runtime_ops_table_in_sync.py (table)
// and tests/unit/test_apple_gpu_tile_pass_status_matches_envelope.py
// (real tessera-opt pass behavior).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_APPLE_GPU_RUNTIME_OP
#error "Define TESSERA_APPLE_GPU_RUNTIME_OP(name) before including this file"
#endif

"""

FOOTER = """\

// Total runtime-executable ops: {total}
#undef TESSERA_APPLE_GPU_RUNTIME_OP
"""


def generate() -> str:
    by_lane: dict[str, list[str]] = defaultdict(list)
    for op in sorted(_APPLE_GPU_RUNTIME_OPS):
        by_lane[APPLE_GPU_LANE_BY_OP[op]].append(op)
    body: list[str] = []
    for lane in sorted(by_lane):
        ops = by_lane[lane]
        body.append(f"// lane: {lane} ({len(ops)} ops)")
        body.extend(f'TESSERA_APPLE_GPU_RUNTIME_OP("{op}")' for op in ops)
        body.append("")
    return HEADER + "\n".join(body) + FOOTER.format(total=len(_APPLE_GPU_RUNTIME_OPS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail (exit 1) if the on-disk .inc is stale")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"STALE: {OUT} — regenerate with "
                  "`python scripts/generate_apple_runtime_ops_table.py`")
            return 1
        print(f"OK: {OUT} in sync ({len(_APPLE_GPU_RUNTIME_OPS)} ops)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(_APPLE_GPU_RUNTIME_OPS)} ops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
