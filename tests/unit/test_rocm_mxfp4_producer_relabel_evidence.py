"""The MXFP4 producer relabel changed no timed gfx1201 kernel.

Four sealed timing packets pin the generator hashes that produced them. The
relabel packet proves, on the owning device, that today's generators build the
same kernels: whole-payload identity for the deterministic WMMA builds and
selected-symbol instruction-stream identity for folded/packed/TN4 builds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/baselines"
PROOF = BASE / "gfx1201_mxfp4_producer_relabel_20260924"
STREAM_PACKETS = (
    "gfx1201_mxfp4_prefill_sweep_20260923",
    "gfx1201_mxfp4_a_offset32_20260923",
    "gfx1201_mxfp4_prefill_experiments_20260923",
)


def _identity() -> dict:
    return json.loads((PROOF / "identity.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_proof_binds_current_generators_and_checkers() -> None:
    identity = _identity()
    assert identity["architecture"] == "gfx1201"
    for path, digest in identity["relabel_generator_sha256"].items():
        assert _sha(ROOT / path) == digest, path
    for name, digest in identity["checker_sha256"].items():
        assert _sha(PROOF / name) == digest, name


def test_parent_generators_are_the_ones_the_packets_recorded() -> None:
    parent = _identity()["parent_generator_sha256"]
    native = json.loads((BASE / "gfx1201_mxfp4_kstep_prefill_20260922/evidence.json").read_text())
    assert native["source"]["generator_sha256"] == parent["python/tessera/compiler/rocm_mxfp4_native.py"]
    for name in STREAM_PACKETS:
        packet = json.loads((BASE / name / "evidence.json").read_text())
        packed = packet.get("packed_generator_sha256", packet.get("packed_abi_sha256"))
        assert packed == parent["python/tessera/compiler/rocm_mxfp4_packed_folded.py"], name
        if name != "gfx1201_mxfp4_a_offset32_20260923":  # v6 predates the folded generator
            assert packet["generator_sha256"] == parent["python/tessera/compiler/rocm_mxfp4_folded.py"], name


def test_every_rebuilt_variant_kept_its_instruction_stream() -> None:
    variants = _identity()["instruction_streams"]["variants"]
    assert len(variants) == 39
    for name, row in variants.items():
        assert row["parent_stream_sha256"] == row["relabel_stream_sha256"], name
        assert row["parent_producer"] == "tessera-lower-to-rocm", name
        assert row["relabel_producer"] == "hand-emitted-hip", name


def test_every_recorded_timed_kernel_is_reproduced() -> None:
    identity = _identity()
    rebuilt = {row["relabel_stream_sha256"] for row in identity["instruction_streams"]["variants"].values()}
    payloads = {row["recorded_image_sha256"] for row in identity["production_wmma_payloads"]["rows"]
                if row["rebuilt_twice_identical"]}
    for name in STREAM_PACKETS:
        packet = json.loads((BASE / name / "evidence.json").read_text())
        for row in packet["rows"]:
            meta = row.get("metadata") or {}
            stream = (meta.get("selected_isa") or {}).get("instruction_stream_sha256")
            if stream is not None:
                assert stream in rebuilt, (name, row["case"], row["engine"])
            elif "image_sha256" in meta:  # WMMA rows carry no stream; their payload is deterministic
                assert meta["image_sha256"] in payloads, (name, row["case"], row["engine"])
    native = json.loads((BASE / "gfx1201_mxfp4_kstep_prefill_20260922/evidence.json").read_text())
    for row in native["rows"]:
        if row["engine"] == "tessera":
            assert row["metadata"]["image_sha256"] in payloads, row["case"]
