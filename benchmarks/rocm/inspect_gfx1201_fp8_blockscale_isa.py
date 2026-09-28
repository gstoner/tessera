#!/usr/bin/env python3
"""ISA census of gfx1201 W8A8 block-scale kernels (ROCM-FP8-BLOCKSCALE-1).

Sync GFX1201-PERF-2026-09-27. Compiles each named variant through the
production route (``lower_blockscale`` -> ``package_blockscale``), with the
same carrier rewrite seam the timing harness uses for sweep-only variants,
disassembles the HSACO with the matched LLVM tools and records the kernel's
resource notes (VGPR/SGPR/LDS/scratch/spills) and a mnemonic census. Static
counts only: nothing here is a runtime or profiler measurement.

    python benchmarks/rocm/inspect_gfx1201_fp8_blockscale_isa.py \\
        --variant 1024,4096,7168:prod --variant 1024,4096,7168:reg:32x32 \\
        --output /tmp/isa.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tessera.compiler.rocm_fp8_blockscale import (  # noqa: E402
    BlockScaleProgram,
    BlockScaleShape,
    lower_blockscale,
    package_blockscale,
)

_NOTES = (".vgpr_count", ".sgpr_count", ".group_segment_fixed_size",
          ".private_segment_fixed_size", ".vgpr_spill_count", ".sgpr_spill_count")
_CENSUS = ("wmma", "global_load", "global_store", "ds_load", "ds_store", "s_barrier",
           "s_wait", "scratch", "global_inv")


def _rewrite(program: BlockScaleProgram, *, staging: str, warps: int,
             macro: tuple[int, int]) -> BlockScaleProgram:
    tile = program.tile_ir
    for pattern, value in ((r'staging = "\w+"', f'staging = "{staging}"'),
                           (r"(?<![\w.])warps = \d+", f"warps = {warps}"),
                           (r"tessera\.macro_tile_m = \d+", f"tessera.macro_tile_m = {macro[0]}"),
                           (r"tessera\.macro_tile_n = \d+", f"tessera.macro_tile_n = {macro[1]}")):
        tile, count = re.subn(pattern, value, tile)
        if count != 1:
            raise SystemExit(f"carrier rewrite did not match {pattern!r}")
    return BlockScaleProgram(program.shape, program.entry, program.graph_ir,
                             program.schedule_ir, tile)


def census(spec: str, llvm_bin: Path) -> dict:
    """``M,N,K:prod[:bf16]`` or ``M,N,K:reg:PMxPN`` or ``M,N,K:lds:MMxMN:W``."""
    shape_text, kind, *rest = spec.split(":")
    m, n, k = (int(v) for v in shape_text.split(","))
    output = "bf16" if "bf16" in rest else "f32"
    shape = BlockScaleShape(m, n, k, 128, 128, "nk", output)
    program = lower_blockscale(shape)
    if kind == "reg":
        macro = tuple(int(v) for v in rest[0].split("x"))
        program = _rewrite(program, staging="global", warps=1, macro=macro)
    elif kind == "lds":
        macro = tuple(int(v) for v in rest[0].split("x"))
        program = _rewrite(program, staging="lds", warps=int(rest[1]), macro=macro)
    elif kind != "prod":
        raise SystemExit(f"unknown variant kind {kind!r}")
    package = package_blockscale(program)
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
        handle.write(package.image.payload)
        handle.flush()
        asm = subprocess.run([str(llvm_bin / "llvm-objdump"), "-d", "--mcpu=gfx1201", handle.name],
                             capture_output=True, text=True, check=True).stdout
        notes = subprocess.run([str(llvm_bin / "llvm-readelf"), "--notes", handle.name],
                               capture_output=True, text=True, check=True).stdout
    ops = Counter(re.findall(r"^\s+([a-z_][a-z0-9_]*)\s", asm, re.M))
    resources = {}
    for key in _NOTES:
        found = re.search(re.escape(key) + r":\s+(\S+)", notes)
        resources[key.lstrip(".")] = int(found.group(1)) if found else None
    prov = package.descriptor.provenance
    return {
        "variant": spec,
        "physical_route": prov["physical_route"],
        "workgroup": prov["workgroup"],
        "macro_tile": prov["macro_tile"],
        "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "resources": resources,
        "census": {name: count for name, count in sorted(ops.items())
                   if any(token in name for token in _CENSUS)},
        "instructions": sum(ops.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llvm_bin = Path(os.environ.get("TESSERA_LLVM_BIN", ""))
    if not (llvm_bin / "llvm-objdump").is_file():
        raise SystemExit("set TESSERA_LLVM_BIN to the matched LLVM 23 bin directory")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    record = {
        "work_item": "ROCM-FP8-BLOCKSCALE-1", "sync_key": "GFX1201-PERF-2026-09-27",
        "source_commit": commit, "worktree_dirty": dirty,
        "tessera_opt": os.environ.get("TESSERA_OPT"),
        "kind": "static ISA census; not a runtime or profiler measurement",
        "variants": [census(spec, llvm_bin) for spec in args.variant],
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for row in record["variants"]:
        print(row["variant"], row["resources"], row["census"].get("v_wmma_f32_16x16x16_fp8_fp8"))


if __name__ == "__main__":
    main()
