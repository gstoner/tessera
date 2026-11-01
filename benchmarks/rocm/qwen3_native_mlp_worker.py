"""Native MLP worker for the stage-2 Qwen3-8B perplexity gate (gfx1201).

Runs in the Tessera project environment and serves MLP calls to the Torch
reference driver (``record_qwen3_native_mlp_perplexity``) over a framed pipe,
so Torch and Tessera never share a process (Decision #23). Two MLP arms:

* ``native``: the public seven-member quantized MLP
  (``canonical_rocm_quantized_mlp_program``), in 128-row chunks; the staged
  tolerance certificate runs on the first chunk of every certified layer;
* ``shipped``: the decoded shipped NVFP4 weights under the stage-1 FP64
  reference policy (``layer_reference``), no activation quantization.

Every output must be finite. Framing: a 4-byte little-endian header length,
the JSON header, then ``header["nbytes"]`` payload bytes (BF16 bit patterns).
The module imports Tessera only inside ``serve`` so the framing helpers stay
importable from the Torch environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np

HIDDEN = 4096
CHUNK = 128


def write_frame(stream: BinaryIO, header: dict, payload: bytes = b"") -> None:
    header = {**header, "nbytes": len(payload)}
    encoded = json.dumps(header).encode()
    stream.write(struct.pack("<I", len(encoded)) + encoded + payload)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if data is None or len(data) != size:
        raise EOFError(f"pipe closed after {0 if data is None else len(data)} of {size} bytes")
    return data


def read_frame(stream: BinaryIO) -> tuple[dict, bytes]:
    (size,) = struct.unpack("<I", _read_exact(stream, 4))
    if not 0 < size < 1 << 24:
        raise ValueError(f"implausible frame header size {size}")
    header = json.loads(_read_exact(stream, size))
    nbytes = header.get("nbytes")
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("frame header carries no byte count")
    return header, _read_exact(stream, nbytes)


def _rows(payload: bytes, rows: int) -> np.ndarray:
    import ml_dtypes

    if rows <= 0 or rows % CHUNK or len(payload) != rows * HIDDEN * 2:
        raise ValueError(f"MLP call needs a multiple of {CHUNK} rows of {HIDDEN} BF16 values")
    return np.frombuffer(payload, np.uint16).reshape(rows, HIDDEN).view(ml_dtypes.bfloat16)


def serve(cache_root: Path, certify: str, inbound: BinaryIO, outbound: BinaryIO) -> None:
    import ml_dtypes
    import tessera as ts
    from tessera import runtime as rt

    from benchmarks.rocm import qwen3_local_checkpoint as local
    from benchmarks.rocm.benchmark_checkpoint_quantized_mlp import decode_projection, layer_reference
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp
    from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
    from benchmarks.rocm.record_qwen3_mlp_layer_quality import _device_identity

    uuid, name = _device_identity(rt)
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise RuntimeError("native ROCm movement runtime unavailable")
    snapshots = (local.LocalSnapshot.open(cache_root, local.BF16_REPO),
                 local.LocalSnapshot.open(cache_root, local.NVFP4_REPO))
    fn, layer, fixed, decoded, certified = None, None, None, None, False
    certificates: dict[int, object] = {}
    calls = {"native": 0, "shipped": 0}
    while True:
        header, payload = read_frame(inbound)
        op = header.get("op")
        if op == "hello":
            write_frame(outbound, {
                "ok": True, "architecture": "gfx1201", "device_uuid": uuid, "device_name": name,
                "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
                "provider_sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()})
        elif op == "layer":
            layer = header["layer"]
            loaded = local.load_layer_mlp(cache_root, layer, bf16=snapshots[0], nvfp4=snapshots[1])
            gate, up, down = loaded["projections"]
            fixed = [np.ascontiguousarray(np.concatenate([gate.packed_codes, up.packed_codes])),
                     np.ascontiguousarray(np.concatenate([gate.e4m3_scales, up.e4m3_scales])),
                     np.array([gate.global_scale, up.global_scale], np.float64),
                     down.packed_codes, down.e4m3_scales, np.array([down.global_scale], np.float64)]
            decoded = [decode_projection(p) for p in loaded["projections"]]
            if fn is None:
                h, k, n = gate.packed_codes.shape[0], gate.packed_codes.shape[1] * 2, down.packed_codes.shape[0]
                if k != HIDDEN or n != HIDDEN:
                    raise ValueError("checkpoint geometry differs from the worker's hidden size")
                fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(CHUNK, h, k, n))
            certified = certify == "none" or (certify == "first-layer" and bool(certificates))
            write_frame(outbound, {"ok": True, "layer": layer, "source": loaded["source"]})
        elif op == "mlp":
            if fixed is None or header.get("layer") != layer:
                raise RuntimeError("MLP call for a layer the worker has not loaded")
            x = _rows(payload, header["rows"])
            out = np.empty_like(x)
            for start in range(0, x.shape[0], CHUNK):
                chunk = np.ascontiguousarray(x[start:start + CHUNK])
                if header["arm"] == "native":
                    roots = [chunk, *fixed]
                    result = fn(*roots)
                    artifact = fn.runtime_artifact()
                    if fn.execution_kind != "native_gpu" \
                            or artifact.metadata["compiler_path"] != "canonical_rocm_quantized_mlp_program":
                        raise RuntimeError("public call did not execute the seven-member native program")
                    if not certified:
                        certificates[layer] = assert_compiled_chain(
                            fn._rocm_quantized_product_last_program.native, roots, result, lib._name)
                        certified = True
                elif header["arm"] == "shipped":
                    result = layer_reference(chunk, *decoded)
                else:
                    raise ValueError(f"unknown MLP arm {header['arm']!r}")
                result = np.asarray(result).astype(ml_dtypes.bfloat16)
                if not np.isfinite(result.astype(np.float32)).all():
                    raise RuntimeError(f"layer {layer} {header['arm']} rows {start}.. produced non-finite output")
                out[start:start + CHUNK] = result
                calls[header["arm"]] += 1
            write_frame(outbound, {"ok": True}, np.ascontiguousarray(out).view(np.uint16).tobytes())
        elif op == "close":
            write_frame(outbound, {"ok": True, "chunk_calls": calls, "certify": certify,
                                   "certificates": {str(k): v for k, v in sorted(certificates.items())}})
            return
        else:
            raise ValueError(f"unknown worker op {op!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--certify", choices=("every-layer", "first-layer", "none"), default="every-layer")
    args = parser.parse_args()
    # The pipe owns stdout; anything the stack prints goes to stderr instead.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    serve(args.cache_root, args.certify, sys.stdin.buffer, protocol_out)
