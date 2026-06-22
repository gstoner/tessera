#!/usr/bin/env python3
"""Sprint H-2 (2026-05-11) — Generate `mfma_table.inc` from
`tessera.compiler.rocm_target._MFMA_VARIANTS`.

The C++ ROCm backend consumes `mfma_table.inc` as a static include of
per-arch MFMA instruction shapes.  Keeping it in sync with the Python
`_MFMA_VARIANTS` source is enforced by ``test_mfma_table_in_sync.py``:
the generator is the single source of truth for shape tuples; the
inline `.inc` file mirrors them in X-macro form so a `#include` brings
in all variants at once.

Run:
    python scripts/generate_mfma_table.py

Or as part of CI / pre-commit:
    python scripts/generate_mfma_table.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler.rocm_target import (
    AMDArch,
    TESSERA_TARGET_ROCM,
    TESSERA_TARGET_HIP,
    mfma_variants,
)


HEADER = """\
//===- mfma_table.inc - AMD MFMA instruction shape table -----*- C++ -*-===//
//
// Auto-generated from `python/tessera/compiler/rocm_target.py`
// _MFMA_VARIANTS.  Do NOT edit by hand — regenerate with:
//
//     python scripts/generate_mfma_table.py
//
// X-macro format: each line is a TESSERA_MFMA_VARIANT call that the
// includer expands.  Fields: (arch_id, arch_name, M, N, K, K_blocks).
//
// Sync gate: `tests/unit/test_mfma_table_in_sync.py` fails if this file
// disagrees with the Python source.
//
// ROCm target pin: {rocm}
// HIP target pin:  {hip}
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_MFMA_VARIANT
#error "Define TESSERA_MFMA_VARIANT(arch_id, arch_name, M, N, K, K_blocks) before including this file"
#endif

"""

FOOTER = """\

// Total shapes across all ROCm 7.2.4 arches: {total}
#undef TESSERA_MFMA_VARIANT
"""


def _emit_arch_block(arch: AMDArch) -> str:
    """Emit X-macro lines for a single arch's MFMA shapes."""
    shapes = sorted(mfma_variants(arch))
    if not shapes:
        return f"// {arch.name} ({arch.value}) — no MFMA (WMMA-only or unsupported)\n"
    lines = [f"// {arch.name} ({arch.value}) — {len(shapes)} MFMA shapes"]
    for (m, n, k, k_blocks) in shapes:
        lines.append(
            f"TESSERA_MFMA_VARIANT({arch.value}, \"{arch.name.lower()}\", "
            f"{m}, {n}, {k}, {k_blocks})"
        )
    return "\n".join(lines) + "\n"


def generate() -> str:
    """Return the full `.inc` text."""
    total = 0
    body = []
    for arch in AMDArch:
        body.append(_emit_arch_block(arch))
        total += len(mfma_variants(arch))
    return (
        HEADER.format(rocm=TESSERA_TARGET_ROCM, hip=TESSERA_TARGET_HIP)
        + "\n".join(body)
        + FOOTER.format(total=total)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify the on-disk file matches the generated content; exit 1 on drift.",
    )
    args = ap.parse_args()

    out_path = (
        ROOT
        / "src" / "compiler" / "codegen" / "Tessera_ROCM_Backend"
        / "include" / "TesseraROCM" / "mfma_table.inc"
    )
    new_text = generate()

    if args.check:
        if not out_path.exists():
            print(f"ERROR: {out_path} missing (run without --check to create)")
            return 1
        on_disk = out_path.read_text()
        if on_disk != new_text:
            print(f"ERROR: {out_path} drifted from _MFMA_VARIANTS source.")
            print("       Regenerate with: python scripts/generate_mfma_table.py")
            return 1
        print(f"OK: {out_path} in sync")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_text)
    print(f"Wrote {out_path} ({len(new_text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
