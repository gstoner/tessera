#!/usr/bin/env python3
"""Exact-operation packet for the compiler-emitted sm_120a NVFP4 block-scale
warp tile (`ptx_emit.emit_nvfp4_block_scale_mma_ptx`) on the owning device.

For each scale mode the on-silicon spike ran (unit, uniform 0.5 and 2.0,
mapped non-uniform) plus a random-scale row, records the max absolute error
of the device tile against `nvfp4_fragments.nvfp4_tile_reference`, the entry
name, the PTX digest and the `.version` the driver JIT was handed. One fixed
m16n8k64 tile per launch: correctness evidence for the emitted kernel and
its per-lane packer, no performance claim (per-call host transfers, no timer).

Usage (Super-Bear, `scripts/_nvidia_env.sh` sourced, build-nvidia-cuda built)::

    python benchmarks/record_nvfp4_emitted.py --output \\
        benchmarks/baselines/nvfp4_emitted_20260918/nvidia_sm120.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import nvfp4_fragments as nf  # noqa: E402
from tessera.compiler import ptx_emit as pe  # noqa: E402
from tessera.compiler.gpu_target import ptx_for_driver_jit  # noqa: E402


def scales(mode):
    codes = np.array([0x30, 0x38, 0x40], dtype=np.uint8)
    sa = np.full((nf.M, nf.SCALE_BLOCKS), nf.UE4M3_ONE, np.uint8)
    sb = np.full((nf.SCALE_BLOCKS, nf.N), nf.UE4M3_ONE, np.uint8)
    if mode == "unit":
        return sa, sb
    if mode.startswith("uniform:"):
        code = int(mode.split(":")[1], 16)
        return np.full_like(sa, code), np.full_like(sb, code)
    if mode == "mapped":
        rows, blocks = np.indices(sa.shape)
        sa = codes[(rows + blocks) % 3]
        blocks, cols = np.indices(sb.shape)
        sb = codes[(2 * blocks + cols) % 3]
        return sa, sb
    rng = np.random.default_rng(9)
    return (rng.integers(0x28, 0x48, size=sa.shape, dtype=np.uint8),
            rng.integers(0x28, 0x48, size=sb.shape, dtype=np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if rt._load_nvidia_ptx_launch() is None:
        raise SystemExit("libtessera_nvidia_ptx_launch.so is not built on this host")
    ptx = pe.emit_nvfp4_block_scale_mma_ptx()
    jit_text, lowered_from = ptx_for_driver_jit(ptx)
    rows = []
    for mode in ("unit", "uniform:30", "uniform:40", "mapped", "random"):
        rng = np.random.default_rng(11)
        a = rng.integers(0, 16, size=(nf.M, nf.K), dtype=np.uint8)
        b = rng.integers(0, 16, size=(nf.K, nf.N), dtype=np.uint8)
        sa, sb = scales(mode)
        out = rt._nvidia_nvfp4_emitted_mma(a, b, sa, sb)
        ref = nf.nvfp4_tile_reference(a, b, sa, sb)
        error = float(np.max(np.abs(out - ref)))
        if error != 0.0:
            raise SystemExit(f"{mode}: device tile disagrees with the exact reference (max abs error {error})")
        rows.append(dict(scale_mode=mode, max_abs_error=error, elements=int(out.size)))
    packet = dict(
        schema="tessera.nvfp4_emitted.v1",
        backend="nvidia", chip="sm_120", arch="sm_120a", host=platform.platform(),
        entry=pe.TESSERA_NVFP4_MMA_ENTRY, mnemonic=pe.NVFP4_MMA_MNEMONIC,
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
        driver_jit_ptx_version=jit_text.split("\n", 1)[0].split()[-1],
        ptx_version_lowered_from=lowered_from,
        rows=rows,
        evidence_scope="exact_device",
        promotion=dict(correctness_eligible=True, performance_eligible=False,
                       reason="one fixed m16n8k64 warp tile per launch with host transfers; no general-shape dispatch, no timer"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(rows)} rows, all exact)")


if __name__ == "__main__":
    main()
