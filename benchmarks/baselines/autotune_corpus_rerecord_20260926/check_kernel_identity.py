"""Kernel-code identity across build trees, and corpus rows served in a tree
that did not record them (Decision #11, sync AUTOTUNE-TOOLCHAIN-KEY-2026-09-26).

Run on Princess-Luna (gfx1151) from the repo root with the ROCm env sourced.
The build tree is chosen with ``TESSERA_BUILD_DIR`` (``runtime._tessera_opt_path``).

  identity  -- for each of the 8 gfx1151 fused_region keys, the tessera-opt
               binary digest, the fused image's payload and per-section
               digests, and the kernel-code identity. JSON on stdout.
  compare   -- two ``identity`` outputs: which fields differ between the trees.
  served    -- load the committed corpus and ask for each key the way
               production does (no explicit dims): ordinary ``run_arbitrated``
               dispatch, ``corpus_winner``, and ``measured_arbitrate`` with
               re-measurement disabled (a miss raises instead of timing).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys

import numpy as np

SHAPES = (64, 256, 512, 1024)


def _sections(payload: bytes) -> dict[str, str]:
    """sha256 of every ELF64 section's bytes, by name (NOBITS: its size)."""
    shoff, = struct.unpack_from("<Q", payload, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", payload, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", payload, shoff + i * shentsize)
               for i in range(shnum)]
    strtab = headers[shstrndx]
    names = payload[strtab[4]:strtab[4] + strtab[5]]
    out: dict[str, str] = {}
    for name_off, kind, _flags, _addr, off, size, *_ in headers[1:]:
        name = names[name_off:names.index(b"\0", name_off)].decode()
        data = b"" if kind == 8 else payload[off:off + size]
        out[name] = hashlib.sha256(data).hexdigest()[:16] if kind != 8 else f"nobits:{size}"
    return out


def _workload(size: int):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    bias = rng.standard_normal((size,)).astype(np.float32)
    return a, b, bias


def identity() -> None:
    from tessera import runtime as rt
    from tessera.compiler import fusion as F
    from tessera.compiler import kernel_code_identity as KI
    from tessera.compiler.emit import rocm_hip
    from tessera.compiler.toolchain_identity import tessera_opt_identity

    region = F.FusedRegion(epilogue=("bias", "gelu"))
    cand = rocm_hip.RocmWmmaGemmCandidate()
    out = {"tessera_opt": str(rt._tessera_opt_path()),
           "tessera_opt_identity": tessera_opt_identity(),
           "objdump": KI.find_llvm_objdump(), "keys": {}}
    for size in SHAPES:
        a, b, bias = _workload(size)
        payload, entry = rt._rocm_wmma_fused_image(size, size, size, "f16",
                                                   bias=True, activation="gelu")
        out["keys"][str(size)] = {
            "entry": entry,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "sections": _sections(payload),
            "kernel_identity": cand.artifact_identity(region, a, b, bias),
        }
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    print()


def compare(path_a: str, path_b: str) -> int:
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    print(f"tree A tessera-opt {a['tessera_opt']}\n  abi_digest {a['tessera_opt_identity']['abi_digest']}")
    print(f"tree B tessera-opt {b['tessera_opt']}\n  abi_digest {b['tessera_opt_identity']['abi_digest']}")
    print(f"tessera-opt binaries identical: "
          f"{a['tessera_opt_identity']['abi_digest'] == b['tessera_opt_identity']['abi_digest']}")
    bad = 0
    for size in map(str, SHAPES):
        ka, kb = a["keys"][size], b["keys"][size]
        varied = sorted(s for s in set(ka["sections"]) | set(kb["sections"])
                        if ka["sections"].get(s) != kb["sections"].get(s))
        same_kid = ka["kernel_identity"] == kb["kernel_identity"] and ka["kernel_identity"]
        bad += not same_kid
        kid = ka["kernel_identity"] or {}
        print(f"{size}^3 f16 bias+gelu: payload identical={ka['payload_sha256'] == kb['payload_sha256']}"
              f" sections varied={varied or 'none'} kernel identity identical={bool(same_kid)}"
              f" stream={kid.get('instruction_stream_sha256', 'NONE')[:16]}"
              f" kd={kid.get('kernel_descriptor_sha256', 'NONE')[:16]}"
              f" n={kid.get('instruction_count')} data={kid.get('data_sections')}"
              f" disassembler={kid.get('disassembler')!r}")
    return 1 if bad else 0


def served() -> int:
    """Each committed row, asked for the way production asks: no explicit dims.

    * ``run_arbitrated(region, op, "rocm", a, b, bias)`` -- ordinary dispatch.
      ``corpus_winner`` is wrapped only to record what it returned; dispatch
      consults the device row first, so this shows the device row serving.
    * ``corpus_winner`` for each timing, dims inferred.
    * ``measured_arbitrate`` for each timing, dims inferred, with its miss path
      (``arbitrate``) replaced by a raise: a returned winner came from the row.
    """
    from tessera.compiler import fusion as F
    from tessera.compiler import kernel_code_identity as KI
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit import candidate as C
    from tessera.compiler.emit import rocm_hip  # noqa: F401 - registers candidates
    from tessera.compiler.emit.candidate import OP_FUSED_REGION
    from tessera.compiler.toolchain_identity import tessera_opt_identity

    region = F.FusedRegion(epilogue=("bias", "gelu"))
    print(f"serving tessera-opt abi_digest {tessera_opt_identity()['abi_digest']}; "
          f"device key {at._device_id('rocm')}; disassembler {KI.find_llvm_objdump()}")
    committed = at.MeasureCache()
    at.load_corpus(cache=committed)
    real_corpus_winner = at.corpus_winner
    real_arbitrate = at.arbitrate
    bad = 0
    for size in SHAPES:
        a, b, bias = _workload(size)
        dims = at._infer_dims(OP_FUSED_REGION, (a, b, bias))
        live = rocm_hip.RocmWmmaGemmCandidate().artifact_identity(region, a, b, bias)
        recs = {}
        for timing in (at.TIMING_END_TO_END, at.TIMING_DEVICE):
            key = ("rocm:gfx1151", "rocm", OP_FUSED_REGION,
                   at.bucket_key(dims, at.SpecPolicy.BUCKET), "f16", timing)
            recs[timing] = committed._store.get(key)

        # 1. ordinary dispatch
        answers: list[tuple[str, object]] = []

        def recording(*args, **kwargs):
            result = real_corpus_winner(*args, **kwargs)
            answers.append((kwargs.get("timing"), result))
            return result

        at.corpus_winner = recording
        C.reset_arbiter_dispatch_log()
        try:
            _, tag = C.run_arbitrated(region, OP_FUSED_REGION, "rocm", a, b, bias)
        finally:
            at.corpus_winner = real_corpus_winner
        selected = C.arbiter_dispatch_log()[-1][2]
        device_rec = recs[at.TIMING_DEVICE]
        dispatch_ok = (device_rec is not None and answers
                       and answers[0] == (at.TIMING_DEVICE, device_rec.winner)
                       and selected == device_rec.winner and tag != "reference")
        print(f"{size}^3 dims={dims} run_arbitrated: corpus_winner answers={answers} "
              f"selected={selected} tag={tag} -> "
              f"{'SERVED (device row)' if dispatch_ok else 'NOT SERVED'}")
        bad += not dispatch_ok

        # 2./3. each row, both lookups, dims inferred
        for timing, rec in recs.items():
            recorded = (rec.evidence.get("delegate_identities", {}).get("rocm_wmma_gemm")
                        if rec else None)
            cache = at.MeasureCache()
            at.load_corpus(cache=cache)
            hint = at.corpus_winner(region, OP_FUSED_REGION, "rocm", a, b, bias,
                                    dtype="f16", cache=cache, timing=timing)

            def refuse(*args, **kwargs):
                raise RuntimeError("re-measure attempted: the row was NOT served")

            at.arbitrate = refuse
            try:
                chosen = at.measured_arbitrate(
                    region, OP_FUSED_REGION, "rocm", a, b, bias,
                    dtype="f16", cache=cache, timing=timing).name
            except RuntimeError as exc:
                chosen = f"MISS ({exc})"
            finally:
                at.arbitrate = real_arbitrate
            ok = rec is not None and hint == rec.winner and chosen == rec.winner
            bad += not ok
            sep = (rec.separation or {}) if rec else {}
            print(f"    {timing:10s} recorded={rec.winner if rec else None}"
                  f" separated={sep.get('separated')}"
                  f" identity_mismatch={KI.identity_mismatch(recorded, live)}"
                  f" corpus_winner={hint} measured_arbitrate={chosen}"
                  f" -> {'SERVED' if ok else 'NOT SERVED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "identity":
        identity()
    elif mode == "compare":
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
    elif mode == "served":
        raise SystemExit(served())
    else:
        raise SystemExit(__doc__)
