#!/usr/bin/env python3
"""Record the MXFP8 native compiler boundary; checked runtime ABI is still open."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tests.device.rocm import test_mxfp8_scheduled_scale as proof
from tests._support import rocm_isa


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("this recorder requires the selected HIP device to be gfx1201")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mnk in [(17, 19, 64), (16, 16, 128), (16, 2, 64)]:
        for layout in ["kn", "nk"]:
            for output in ["f32", "bf16"]:
                shape = proof.BlockScaleShape(
                    *mnk, scale_k=32, scale_n=1, weight_layout=layout, output=output)
                graph, schedule, tile, target, image = proof._compile(shape)
                rocm_isa.assert_selected(
                    image, chip="gfx1201", pattern=r"v_wmma_f32_16x16x16_\w+",
                    require="v_wmma_f32_16x16x16_fp8_fp8", what="native MXFP8 integration")
                entry = proof._directive_symbol(target, "tessera_rocm.scaled_wmma_gemm")
                a, b, sa, sb = proof._inputs(shape)
                expected = proof._reference(a, b, sa, sb)
                actual = proof._launch(image, entry, shape, a, b, sa, sb)
                if output == "bf16":
                    expected = expected.astype(ml_dtypes.bfloat16)
                    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
                else:
                    np.testing.assert_array_equal(actual, expected)
                key = "x".join(map(str, mnk)) + "-" + layout + "-" + output
                stages = dict(graph=graph, schedule=schedule, tile=tile, target=target)
                for stage, text in stages.items():
                    (out / (key + "-" + stage + ".mlir")).write_text(text)
                rows.append(dict(
                    shape_mnk=list(mnk), layout=layout, output=output, entry=entry,
                    numerical="exact_output",
                    finite_elements=int(np.isfinite(actual.astype(np.float32)).sum()),
                    image_sha256=hashlib.sha256(image).hexdigest(),
                    ir_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in stages.items()}))
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    ordinal = ctypes.c_int()
    assert hip.hipGetDevice(ctypes.byref(ordinal)) == 0
    name = ctypes.create_string_buffer(256)
    hip.hipDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    assert hip.hipDeviceGetName(name, len(name), ordinal.value) == 0
    files = [
        "src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/programming_model/ir/ScheduleDialect.cpp",
        "src/compiler/ir/TileDialect.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        "python/tessera/compiler/rocm_mxfp8_blockscale.py",
        "tests/device/rocm/test_mxfp8_scheduled_scale.py",
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    packet = dict(
        schema="tessera.mxfp8_native_schedule.v1", work_item="ROCM-FP8-BLOCKSCALE-1",
        sync_key="ROCM-MXFP8-SCHEDULE-2026-10-03", host=platform.node(),
        architecture=rt._rocm_live_arch(), device=name.value.decode(),
        device_ordinal=ordinal.value,
        compiler_sha256=hashlib.sha256(Path(proof.find_tessera_opt()).read_bytes()).hexdigest(),
        route="textual Graph -> native Schedule -> typed Tile -> ROCm Target -> ROCDL/LLVM -> HSACO -> raw diagnostic HIP",
        checked_runtime_package_proof=False, canonical_mxfp8_storage_promoted=False,
        performance_claim=False,
        source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        rows=rows)
    (out / "numerical.json").write_text(json.dumps(packet, indent=2) + "\n")
    print(f'{packet["device"]}: {len(rows)} native compiler cases passed')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
