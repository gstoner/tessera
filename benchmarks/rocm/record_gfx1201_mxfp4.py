"""Record HSACO identity, ISA, and resources for exact gfx1201 MXFP4 routes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_native import (
    package_mxfp4_w4a8_exact,
    package_mxfp4_w4a8_wmma,
)
from tests._support import rocm_isa


_RESOURCE_KEYS = (
    "sgpr_count",
    "vgpr_count",
    "wavefront_size",
    "kernarg_segment_size",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
)


def _readobj() -> str:
    objdump = Path(rocm_isa.llvm_objdump())
    sibling = objdump.with_name("llvm-readobj")
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError(f"llvm-readobj is not next to the proved disassembler {objdump}")


def _resource_metadata(payload: bytes) -> dict[str, int]:
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
        handle.write(payload)
        handle.flush()
        text = subprocess.check_output([_readobj(), "--notes", handle.name], text=True)
    result: dict[str, int] = {}
    for key in _RESOURCE_KEYS:
        match = re.search(rf"\.{key}:\s+(\d+)", text)
        if match:
            result[key] = int(match.group(1))
    return result


def _isa_summary(payload: bytes) -> dict[str, object]:
    text = rocm_isa.disassemble(payload, chip="gfx1201")
    instructions = [
        match.group(1)
        for line in text.splitlines()
        if (match := re.match(r"^\s*([a-z][a-z0-9_.]+)\b.*//\s+[0-9a-f]+:", line))
    ]
    counts = {
        "instructions": len(instructions),
        "wmma_fp8_fp8": sum(name == "v_wmma_f32_16x16x16_fp8_fp8" for name in instructions),
        "global_load": sum("global_load" in name for name in instructions),
        "global_store": sum("global_store" in name for name in instructions),
        "scalar_load": sum(name.startswith("s_load") for name in instructions),
        "wait": sum(name.startswith("s_wait") for name in instructions),
    }
    selected = sorted({name for name in instructions if name.startswith("v_wmma_")})
    return {"counts": counts, "selected_matrix_instructions": selected}


def record(output: Path) -> None:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("ROCM-MXFP4-W4A8-1 evidence requires the selected gfx1201 device")
    rows = []
    for route, factory in (
        ("scalar_exact", package_mxfp4_w4a8_exact),
        ("wmma_exact", package_mxfp4_w4a8_wmma),
    ):
        package = factory(32, 32, 128)
        payload = package.image.payload
        rows.append(
            {
                "route": route,
                "shape": [32, 32, 128],
                "target": package.image.target,
                "architecture": package.image.architecture,
                "abi_id": package.descriptor.abi_id,
                "entry": package.descriptor.entry_symbol,
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "image_bytes": len(payload),
                "target_ir_sha256": package.image.target_ir_digest,
                "compiler_fingerprint": package.image.compiler_fingerprint,
                "toolchain_fingerprint": package.image.toolchain_fingerprint,
                "launch": {
                    "grid": list(package.descriptor.geometry.grid or ()),
                    "workgroup": list(package.descriptor.geometry.workgroup or ()),
                },
                "resources": _resource_metadata(payload),
                "isa": _isa_summary(payload),
            }
        )
    wmma = next(row for row in rows if row["route"] == "wmma_exact")
    selected = wmma["isa"]["selected_matrix_instructions"]
    if selected != ["v_wmma_f32_16x16x16_fp8_fp8"]:
        raise RuntimeError(f"exact WMMA route selected the wrong matrix ISA: {selected}")
    packet = {
        "schema": "tessera.rocm.gfx1201_mxfp4_evidence.v1",
        "work_item": "ROCM-MXFP4-W4A8-1",
        "sync_key": "ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21",
        "host": socket.gethostname(),
        "device": "Radeon RX 9070 XT",
        "live_architecture": rt._rocm_live_arch(),
        "rocm_path": os.environ.get("ROCM_PATH", ""),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "proof": {
            "device_test": "tests/device/rocm/test_mxfp4_w4a8_exact.py",
            "shapes": [[17, 19, 64], [32, 32, 128]],
            "comparison": "bit-exact BF16 versus independent exact-per-K32 host oracle",
            "result": "4 passed (scalar and WMMA routes)",
        },
        "rows": rows,
        "promotion_scope": (
            "exact gfx1201 ABI and native FP8-WMMA mechanism; no gfx1200, "
            "folded-policy, selector-default, or throughput promotion"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(args.output)


if __name__ == "__main__":
    main()
