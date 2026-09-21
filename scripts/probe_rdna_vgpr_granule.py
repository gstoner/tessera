#!/usr/bin/env python3
"""Resolve the RDNA VGPR file size and allocation granule by observation.

Why infer rather than ask
-------------------------
``rocm_occupancy`` needs two per-arch constants: the physical VGPR file per
SIMD, and the allocation block size.  The RDNA4 ISA (3.3.2.1) prescribes a
granule of 24 for wave32 on devices with 1536 VGPRs per SIMD; the ROCm queue's
one recorded data point (121 VGPRs -> 12 waves/SIMD) implies 16.  Both cannot
hold, and the answer moves every RDNA4 register target by 8 VGPRs.

This probe does not ask any tool for the constants -- no tool reports them
directly, and a tool that did would be one more unverified claim.  It compiles
a family of kernels at varying register pressure, reads the *pair*
``(VGPRs, Occupancy [waves/SIMD])`` that the compiler's own resource-usage
remark prints for each, and solves for the ``(file, granule)`` pair that
explains every observation under::

    waves = file // min(ceil(vgprs / granule) * granule, per_wave_cap)

If exactly one candidate survives, that is the answer.  If several do, the
probe says which register count would separate them.  If none do, the model in
``rocm_occupancy`` is wrong and the observations are printed so the next reader
starts from data rather than from this docstring.

Run it on the box that has the device -- the arch is a compile-time target, so
this needs only a matching hipcc, but the constants are per-part and a result
labelled with the wrong arch is worse than no result.  On Tajasarus::

    source ~/.config/tessera/env.sh && source scripts/_rocm_env.sh
    python3 scripts/probe_rdna_vgpr_granule.py --arch gfx1201 \
        --output benchmarks/baselines/gfx1201_vgpr_granule.json

Exit status is 0 when a unique solution is found, 1 otherwise; an inconclusive
run is not a failure of the host, it is an absence of evidence, and the JSON
says which.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

#: Candidate VGPR file sizes per SIMD across the GCN/RDNA/CDNA line.
#: Kept deliberately wide.  The first gfx1201 run omitted 3072 and still
#: resolved, but "unique over a list that excluded the alternative" is not
#: uniqueness -- a candidate set narrow enough to guarantee a single answer is
#: a way of assuming the conclusion.  Re-solving over this wider set still
#: gives exactly one triple for gfx1201.
CANDIDATE_FILES = (512, 768, 1024, 1536, 2048, 3072, 4096)
#: Candidate wave32 allocation granules.  ISA 3.3.2.1 names 16 and 24; the
#: others are here so an unexpected part produces "no solution" rather than a
#: forced fit into one of the two we expected.
CANDIDATE_GRANULES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

#: Candidate hardware wave slots per SIMD.  Solved for rather than assumed:
#: at low register pressure occupancy plateaus at exactly this number, so the
#: probe measures it for free -- and it is independently worth measuring,
#: because ``rocm_target._MAX_WAVES`` carries 16 under the comment "per CU"
#: while ``rocm_occupancy`` carries 16 per SIMD, and a CU is two SIMD32s.
CANDIDATE_WAVE_SLOTS = (8, 10, 12, 16, 20, 24, 32)

PER_WAVE_CAP = 256

#: Register pressure is forced with __launch_bounds__ plus a live-value chain.
#: A chain rather than an array because an array may be promoted to scratch,
#: which changes the VGPR count without changing the source in a readable way.
KERNEL_TEMPLATE = r"""
#include <hip/hip_runtime.h>

extern "C" __global__ void __launch_bounds__({threads}, {min_waves})
probe_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc[{depth}];
#pragma unroll
  for (int k = 0; k < {depth}; ++k) acc[k] = in[(i + k * 97) % n];
  // Keep every element live across a dependent loop so the allocator cannot
  // collapse the chain.
  for (int t = 0; t < n; ++t) {{
#pragma unroll
    for (int k = 0; k < {depth}; ++k) acc[k] = fmaf(acc[k], acc[({depth} - 1) - k], 1.0f);
  }}
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < {depth}; ++k) s += acc[k];
  out[i] = s;
}}
"""

_VGPR_RE = re.compile(r"\bVGPRs:\s*(\d+)")
_OCC_RE = re.compile(r"Occupancy \[waves/SIMD\]:\s*(\d+)")


def _predict(
    vgprs: int,
    file_size: int,
    granule: int,
    wave_slots: int,
    cap: int = PER_WAVE_CAP,
) -> int:
    """Occupancy a device with these constants would report for *vgprs*.

    The wave-slot ceiling is not optional.  Omit it and every low-pressure
    observation becomes unexplainable: a kernel at 14 VGPRs is limited by
    slots, not registers, so a register-only prediction of 64 waves/SIMD
    contradicts the 16 the compiler reports, and the solver then rejects the
    *correct* device as a contradiction.  That is exactly what the first
    gfx1201 run did.
    """
    allocated = min(math.ceil(vgprs / granule) * granule, cap)
    if not allocated:
        return 0
    return min(file_size // allocated, wave_slots)


def compile_one(
    hipcc: str, arch: str, depth: int, threads: int, min_waves: int
) -> dict[str, Any]:
    """Compile one pressure level and read back (VGPRs, Occupancy)."""
    source = KERNEL_TEMPLATE.format(depth=depth, threads=threads, min_waves=min_waves)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.hip"
        src.write_text(source)
        cmd = [
            hipcc,
            f"--offload-arch={arch}",
            "-O3",
            "-c",
            "-Rpass-analysis=kernel-resource-usage",
            str(src),
            "-o",
            str(Path(tmp) / "probe.o"),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return {"depth": depth, "status": "timeout"}
        text = proc.stdout + proc.stderr
        vgprs = _VGPR_RE.search(text)
        occ = _OCC_RE.search(text)
        if proc.returncode != 0 or not (vgprs and occ):
            return {
                "depth": depth,
                "status": "no-remark",
                "returncode": proc.returncode,
                # Truncated: enough to diagnose a toolchain problem, not a dump.
                "diagnostic": text[-2000:],
            }
        return {
            "depth": depth,
            "threads": threads,
            "status": "ok",
            "vgprs": int(vgprs.group(1)),
            "occupancy": int(occ.group(1)),
        }


def solve(observations: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Every (file, granule, wave_slots) consistent with all observations."""
    usable = [o for o in observations if o.get("status") == "ok"]
    if not usable:
        return []
    return [
        (f, g, w)
        for f in CANDIDATE_FILES
        for g in CANDIDATE_GRANULES
        for w in CANDIDATE_WAVE_SLOTS
        if all(_predict(o["vgprs"], f, g, w) == o["occupancy"] for o in usable)
    ]


def separating_register_counts(candidates: list[tuple[int, int, int]]) -> list[int]:
    """Register counts at which the surviving candidates predict different
    occupancies -- i.e. what to measure next to break the tie."""
    if len(candidates) < 2:
        return []
    out = []
    for v in range(1, PER_WAVE_CAP + 1):
        preds = {_predict(v, f, g, w) for f, g, w in candidates}
        if len(preds) > 1:
            out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx1201", help="offload arch to compile for")
    ap.add_argument("--output", type=Path, help="write the JSON packet here")
    ap.add_argument(
        "--threads", type=int, default=256, help="workgroup size for the probes"
    )
    ap.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224],
        help="live-value chain depths; each yields one pressure level",
    )
    args = ap.parse_args()

    hipcc = shutil.which("hipcc")
    packet: dict[str, Any] = {
        "probe": "rdna_vgpr_granule",
        "arch": args.arch,
        "host": platform.node(),
        "platform": platform.platform(),
        "threads": args.threads,
    }

    if hipcc is None:
        packet["status"] = "unavailable"
        packet["reason"] = (
            "hipcc not on PATH; source scripts/_rocm_env.sh on a ROCm host. "
            "This is an absence of evidence, not a negative result."
        )
        _emit(packet, args.output)
        return 1

    packet["hipcc"] = hipcc
    observations = [
        compile_one(hipcc, args.arch, d, args.threads, mw)
        # Vary min_waves too: __launch_bounds__ second argument pushes the
        # allocator onto different rungs, widening the observed VGPR spread.
        # __launch_bounds__ min-waves pushes the allocator onto different
        # rungs; 1 lets it use the whole per-wave budget, widening the spread.
        for d, mw in zip(args.depths, [8, 8, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1])
    ]
    packet["observations"] = observations

    ok = [o for o in observations if o.get("status") == "ok"]
    packet["distinct_vgpr_counts"] = sorted({o["vgprs"] for o in ok})

    if not ok:
        packet["status"] = "inconclusive"
        packet["reason"] = (
            "no kernel produced a kernel-resource-usage remark; check that this "
            "hipcc supports -Rpass-analysis=kernel-resource-usage"
        )
        _emit(packet, args.output)
        return 1

    candidates = solve(observations)
    packet["candidates"] = [
        {"regs_per_simd": f, "granule": g, "wave_slots_per_simd": w}
        for f, g, w in candidates
    ]

    if len(candidates) == 1:
        f, g, w = candidates[0]
        packet["status"] = "resolved"
        packet["regs_per_simd"] = f
        packet["granule_wave32"] = g
        packet["wave_slots_per_simd"] = w
        packet["isa_rule_holds"] = (g == 24) if f == 1536 else None
        packet["apply"] = (
            f"set rocm_occupancy._VGPR_REGS_PER_SIMD[{args.arch}] = "
            f"({f}, Provenance.MEASURED) and, if granule {g} contradicts the "
            f"ISA 3.3.2.1 rule, replace the derivation in vgpr_alloc_granule "
            f"with a measured table and drop {args.arch} from "
            f"VGPR_GRANULE_CONTESTED"
        )
        _emit(packet, args.output)
        return 0

    if candidates:
        sep = separating_register_counts(candidates)
        packet["status"] = "ambiguous"
        packet["separating_vgpr_counts"] = sep[:32]
        packet["reason"] = (
            f"{len(candidates)} (file, granule) pairs explain these "
            f"observations; re-run with a depth that lands near "
            f"{sep[0] if sep else '?'} VGPRs to separate them"
        )
        _emit(packet, args.output)
        return 1

    packet["status"] = "contradiction"
    packet["reason"] = (
        "no (file, granule) pair in the candidate sets explains the observed "
        "(VGPRs, Occupancy) pairs. The occupancy model in rocm_occupancy is "
        "wrong for this part -- start from the observations, not from the model."
    )
    _emit(packet, args.output)
    return 1


def _emit(packet: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(packet, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
