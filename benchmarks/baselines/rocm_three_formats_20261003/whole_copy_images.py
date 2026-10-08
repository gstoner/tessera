"""Record unchanged whole-copy package images; run from the Tessera root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ml_dtypes
import numpy as np
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from tessera.compiler.rocm_mxfp4 import pack_e2m1_codes
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul

parser = argparse.ArgumentParser()
parser.add_argument("--compiler", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["TESSERA_OPT"] = str(args.compiler.resolve())
fp8 = compile_blockscale(BlockScaleShape(200, 4096, 1536, 128, 128, "nk", "bf16"))
mx8 = compile_mxfp8(BlockScaleShape(200, 4096, 1536, 32, 1, "nk", "bf16"))
a = np.ones((256, 128), ml_dtypes.float8_e4m3fn).view(np.uint8)
folded = prepare_folded_weights(pack_e2m1_codes(np.ones((129, 128), np.uint8)),
    np.full((4, 129), 127, np.uint8), allow_approximate=True)
mx4 = compile_folded_scaled_matmul(a, np.ones(256, np.float32), folded,
    tessera_opt=args.compiler, allow_approximate=True).package
args.output.write_text(json.dumps({
    "compiler_sha256": hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
    "packages": {name: {
        "payload_sha256": hashlib.sha256(p.image.payload).hexdigest(),
        "tile_sha256": hashlib.sha256(p.tile_ir.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(p.target_ir.encode()).hexdigest(),
        "abi": p.descriptor.abi_id,
    } for name, p in [("fp8", fp8), ("mxfp8", mx8), ("folded_mxfp4", mx4)]}
}, indent=2) + "\n")
