#!/usr/bin/env python3
"""Static prefill staging census for matched Tessera/Radiance gfx1201 images.

Requested bytes are schedule/source-derived global load bytes before caches,
not measured DRAM traffic. The selected-symbol ISA census is static, not a
dynamic instruction count or phase timer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from tessera import runtime as rt
from tests._support import rocm_isa


ROOT = Path(__file__).resolve().parents[2]
RADIANCE_SYMBOL = (
    "_Z30radiance_mxfp4_fp8_gemm_foldedILi2ELb1ELb1EE"
    "vPKhS1_S1_S1_PKfPDF16biiiiil"
)
_ISA_FAMILIES = (
    "global_load", "ds_load", "ds_store", "s_wait_loadcnt",
    "s_barrier_signal", "s_barrier_wait", "v_wmma", "v_perm_b32",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mnemonics(isa: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in isa.splitlines():
        match = re.match(r"^\s*([a-z][a-z0-9_]+)\b", line)
        if match and match.group(1).startswith(_ISA_FAMILIES):
            counts[match.group(1)] += 1
    return dict(sorted(counts.items()))


def selected_symbol_isa_evidence(payload: bytes, entry_symbol: str) -> dict[str, object]:
    """Fingerprint the selected kernel in the exact HSACO passed to HIP."""
    isa = rocm_isa.disassemble(payload, chip="gfx1201")
    lines = isa.splitlines()
    headers = [
        i for i, line in enumerate(lines)
        if re.fullmatch(rf"[0-9a-f]+ <{re.escape(entry_symbol)}>:", line.strip())
    ]
    if len(headers) != 1:
        raise RuntimeError(
            f"timed HSACO must contain exactly one {entry_symbol} symbol"
        )
    selected: list[str] = []
    for line in lines[headers[0] + 1:]:
        if re.fullmatch(r"[0-9a-f]+ <[^>]+>:", line.strip()):
            break
        if "//" not in line:
            continue
        instruction = " ".join(line.split("//", 1)[0].split())
        if re.match(r"^[a-z][a-z0-9_]+\b", instruction):
            selected.append(instruction)
    if len(selected) < 32:
        raise RuntimeError("selected timed symbol has too few disassembled instructions")
    normalized = "\n".join(selected) + "\n"
    return {
        "entry_symbol": entry_symbol,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "instruction_stream_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "instruction_count": len(selected),
        "mnemonics": _mnemonics(normalized),
    }


def _radiance_selected_isa(module: Path) -> tuple[str, str]:
    objdump = Path(rocm_isa.llvm_objdump())
    with tempfile.TemporaryDirectory(prefix="tessera-radiance-isa-") as directory:
        temporary = Path(directory)
        copied = temporary / module.name
        shutil.copy2(module, copied)
        extracted = subprocess.run(
            [str(objdump), "--offloading", str(copied)],
            cwd=temporary, capture_output=True, text=True, check=True,
        )
        matches = list(temporary.glob(f"{copied.name}.*.hipv4-*gfx1201"))
        if len(matches) != 1:
            raise RuntimeError(
                "pinned Radiance module must contain one gfx1201 image: "
                + extracted.stdout[-400:]
            )
        image = matches[0]
        symbols = subprocess.run(
            [str(objdump.with_name("llvm-nm")), str(image)],
            capture_output=True, text=True, check=True,
        ).stdout
        if RADIANCE_SYMBOL not in symbols:
            raise RuntimeError("pinned Radiance TN2/WPERM/EPIFAST symbol is absent")
        isa = subprocess.run(
            [
                str(objdump), "-d",
                f"--disassemble-symbols={RADIANCE_SYMBOL}", str(image),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.lower()
        if "global_load" not in isa or "v_wmma" not in isa:
            raise RuntimeError("selected Radiance symbol has no load/WMMA ISA")
        return isa, _sha256(image)


def requested_bytes(m: int, n: int, k: int) -> dict[str, int]:
    """Source-derived global bytes for BM256/BN64/BK64, before cache effects."""
    if m <= 64 or n <= 0 or k <= 0 or k % 64:
        raise ValueError("prefill census requires M>64, N>0, K divisible by 64")
    blocks = ((m + 255) // 256) * ((n + 63) // 64)
    a = blocks * 256 * k
    b_folded = blocks * 64 * k
    b_packed = b_folded // 2
    block_scales = b_folded // 32
    return {
        "tessera_a": a,
        "tessera_b_folded": b_folded,
        "radiance_a": a,
        "radiance_b_packed": b_packed,
        "radiance_block_scales": block_scales,
        "radiance_row_reference_minimum": blocks * 64,
    }


def inspect(
    radiance_module: Path, radiance_source: Path, revision: str,
    matched_packet: Path,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("folded prefill ISA census requires selected gfx1201")
    actual_revision = subprocess.run(
        ["git", "-C", str(radiance_source.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if actual_revision != revision:
        raise ValueError("Radiance source revision does not match pinned comparator")
    matched = json.loads(matched_packet.read_text())
    if (
        matched.get("schema") != "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v3"
        or matched.get("radiance", {}).get("wperm") != 1
        or matched.get("radiance", {}).get("binary_sha256") != _sha256(radiance_module)
    ):
        raise ValueError("ISA census requires a layout-verified matched v3 packet")
    cases = ((256, 5120, 8704), (1024, 17408, 5120))
    timed_images: dict[str, dict[str, object]] = {}
    for m, n, k in cases:
        case = f"prefill_{m}x{n}x{k}"
        rows = [
            row for row in matched["rows"]
            if row["case"] == case and row["engine"] == "tessera_folded"
        ]
        if len(rows) != 1:
            raise ValueError(f"ISA census requires one timed folded row for {case}")
        metadata = rows[0]["metadata"]
        selected = metadata.get("selected_isa")
        if not isinstance(selected, dict):
            raise ValueError(f"timed folded row for {case} lacks selected ISA")
        receipt = metadata.get("frontend_receipt") or {}
        if (
            selected.get("payload_sha256") != metadata.get("image_sha256")
            or selected.get("payload_sha256") != receipt.get("hsaco_sha256")
            or selected.get("entry_symbol") != receipt.get("entry_symbol")
        ):
            raise ValueError(f"selected ISA is not bound to timed HSACO for {case}")
        timed_images[case] = {
            "image_sha256": selected["payload_sha256"],
            "entry_symbol": selected["entry_symbol"],
            "instruction_stream_sha256": selected["instruction_stream_sha256"],
            "instruction_count": selected["instruction_count"],
            "isa": selected["mnemonics"],
            "resources": metadata["resources"],
        }
    radiance_isa, radiance_image_sha = _radiance_selected_isa(radiance_module)
    return {
        "schema": "tessera.rocm.gfx1201_folded_staging_census.v2",
        "device": "AMD Radeon RX 9070 XT",
        "architecture": "gfx1201",
        "source_revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "census_sha256": _sha256(Path(__file__)),
        "method": "timed_hsaco_selected_symbol_and_schedule_requested_bytes",
        "not_measured_dram_or_dynamic_instructions": True,
        "tessera": {
            "timed_images": timed_images,
            "generator_sha256": _sha256(
                ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py"
            ),
        },
        "radiance": {
            "source_revision": revision,
            "source_sha256": _sha256(radiance_source),
            "module_sha256": _sha256(radiance_module),
            "image_sha256": radiance_image_sha,
            "selected_symbol": RADIANCE_SYMBOL,
            "isa": _mnemonics(radiance_isa),
            "layout": "fragment_order",
        },
        "matched_packet_sha256": _sha256(matched_packet),
        "requested_bytes": {
            f"{m}x{n}x{k}": requested_bytes(m, n, k) for m, n, k in cases
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-source", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--matched-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = inspect(
        args.radiance_module, args.radiance_source, args.radiance_revision,
        args.matched_packet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
