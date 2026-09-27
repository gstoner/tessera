#!/usr/bin/env python3
"""Recompile the Tessera arms of a recorded W8A8 packet and compare HSACO bytes.

ROCM-FP8-BLOCKSCALE-1, sync GFX1201-PERF-2026-09-27. A timing packet binds the
compiler that produced it. When later commits change the compiler, this check
says which recorded kernels the CURRENT compiler still emits byte for byte --
the only condition under which the old timing still describes the new route.
It launches nothing and times nothing.

    python benchmarks/rocm/check_gfx1201_fp8_blockscale_identity.py \\
        --packet benchmarks/baselines/<dir>/compare.json --arm tessera_nk \\
        --arm tessera_nk+bf16 --output /tmp/identity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tessera.compiler.rocm_fp8_blockscale import (  # noqa: E402
    BlockScaleShape,
    compile_blockscale,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="a production arm label: tessera_<layout>[+bf16]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    rows = []
    for row in packet["rows"]:
        m, n, k = row["shape"]
        for arm in args.arm:
            recorded = row.get(arm, {}).get("hsaco_sha256")
            if recorded is None:
                continue
            layout, _, output = arm.removeprefix("tessera_").partition("+")
            package = compile_blockscale(BlockScaleShape(m, n, k, 128, 128, layout,
                                                         output or "f32"))
            current = hashlib.sha256(package.image.payload).hexdigest()
            rows.append({"shape": [m, n, k], "arm": arm, "recorded": recorded,
                         "current": current, "identical": recorded == current,
                         "physical_route": package.descriptor.provenance["physical_route"]})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    record = {
        "packet": str(args.packet), "packet_source_commit": packet.get("source_commit"),
        "current_commit": commit, "worktree_dirty": dirty,
        "identical": sum(r["identical"] for r in rows), "checked": len(rows), "rows": rows,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{record['identical']}/{record['checked']} recorded kernels are byte-identical "
          f"at {commit[:10]}")


if __name__ == "__main__":
    main()
