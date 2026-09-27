"""Kernel-code identity for compiler-generated autotune candidates (Decision #11).

The arbiter keys a verdict for a ``tessera-opt``-generated kernel on the digest
of the normalized instruction stream of the image it would run for that
workload, not on the compiler binary's digest (which differs on every build).
Host-free: disassembly is a fixture or a fake ``llvm-objdump``; the gfx1151
device proof (two build trees, served in the non-recording tree) is
``benchmarks/baselines/autotune_corpus_rerecord_20260926``.
"""
from __future__ import annotations

import hashlib
import stat
import struct
import textwrap

import numpy as np
import pytest

from tessera.compiler import kernel_code_identity as KI
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit.candidate import (
    OP_MATMUL,
    Candidate,
    Tier,
    register_candidate,
)

LISTING = textwrap.dedent("""\

    /tmp/tmpa1b2c3.hsaco:\tfile format elf64-amdgpu

    Disassembly of section .text:

    0000000000001900 <gemm>:
    \ts_clause 0x5                                               // 000000001900: BF850005
    \ts_load_b64 s[50:51], s[0:1], 0x88                          // 000000001904: F4040C80 F8000088
    \ts_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001944: BF8700C9
    \tv_dual_mov_b32 v81, s9 :: v_dual_and_b32 v92, 15, v0       // 000000001948: CA240009 515C008F
    \ts_cbranch_vccnz 1027                                       // 000000001AB8: BFA40403 <gemm+0x11c8>
    \tv_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7] // 000000001AC0: CC404000 04022108
    \ts_endpgm                                                   // 000000001AC8: BFB00000
""")

# The same kernel as a second build prints it: different load address, a
# different temporary file name, different spacing. Encodings in the comment
# differ too (they are dropped with it).
LISTING_REBUILT = (
    LISTING.replace("/tmp/tmpa1b2c3.hsaco", "/tmp/tmpzz9y8x.hsaco")
    .replace("0000000000001900 <gemm>:", "0000000000002a00 <gemm>:")
    .replace("// 000000001", "// 000000002")
    .replace("s_clause 0x5  ", "s_clause   0x5")
    .replace("<gemm+0x11c8>", "<gemm+0x11c8>  ")
)

KD = textwrap.dedent("""\

    /tmp/tmpa1b2c3.hsaco:\tfile format elf64-amdgpu

    Disassembly of section .rodata:

    00000000000008c0 <gemm.kd>:
    .amdhsa_kernel gemm
    \t.amdhsa_group_segment_fixed_size 0
    \t.amdhsa_private_segment_fixed_size 80
    \t; SHARED_VGPR_COUNT 0
    \t.amdhsa_next_free_vgpr 256
    \t.amdhsa_wavefront_size32 1
    .end_amdhsa_kernel
""")


def _stream_digest(listing: str) -> str:
    text, _ = KI.image_instruction_stream(listing)
    return hashlib.sha256(text.encode()).hexdigest()


def test_two_listings_of_one_kernel_normalize_equal():
    assert _stream_digest(LISTING) == _stream_digest(LISTING_REBUILT)
    text, count = KI.image_instruction_stream(LISTING)
    assert count == 7
    assert text.splitlines()[0] == "<gemm>:"
    # the `//` comment -- address, encoding, branch annotation -- is gone
    assert "bfa40403" not in text and "0x11c8" not in text and "/tmp" not in text
    assert "s_cbranch_vccnz 1027" in text          # PC-relative operand kept


def test_a_changed_instruction_changes_the_digest():
    changed = LISTING.replace("s_cbranch_vccnz 1027", "s_cbranch_vccnz 1026")
    assert _stream_digest(changed) != _stream_digest(LISTING)
    swapped = LISTING.replace("v_wmma_f32_16x16x16_f16", "v_wmma_f32_16x16x16_bf16")
    assert _stream_digest(swapped) != _stream_digest(LISTING)


def test_a_renamed_function_changes_the_digest():
    assert _stream_digest(LISTING.replace("<gemm>:", "<gemm2>:")) != _stream_digest(LISTING)


def test_kernel_descriptor_block_is_normalized_and_sensitive():
    block = KI.kernel_descriptor_block(KD, "gemm")
    assert block.splitlines()[0] == ".amdhsa_kernel gemm"
    assert block.splitlines()[-1] == ".end_amdhsa_kernel"
    assert KI.kernel_descriptor_block(KD.replace("\t.", "    ."), "gemm") == block
    fewer_vgprs = KD.replace("next_free_vgpr 256", "next_free_vgpr 128")
    assert KI.kernel_descriptor_block(fewer_vgprs, "gemm") != block
    with pytest.raises(KI.KernelIdentityUnavailable):
        KI.kernel_descriptor_block(KD, "other")


def test_the_frozen_benchmark_helper_is_the_declared_oracle(monkeypatch):
    """Decision #31: `selected_symbol_isa_evidence` (frozen -- sealed gfx1201
    packets bind its file hash) is the declared oracle for the per-instruction
    normalization. The production digest is built from the same core
    (`instruction_blocks`) -- this checks both halves on one fully decoded
    listing: the core reproduces the oracle's digest, and the production
    stream is exactly that core output under its function header. The two
    differ by design on an undecodable word: the oracle drops it, production
    refuses the image (see the next test)."""
    from benchmarks.rocm import inspect_gfx1201_folded_prefill as oracle
    from tests._support import rocm_isa

    body = "\n".join(
        f"\tv_add_f32_e32 v{i}, v{i}, v{i + 1}   // 0000000019{i:02X}: 0600{i:04X}"
        for i in range(40))
    listing = LISTING.replace("\ts_endpgm", body + "\n\ts_endpgm")
    monkeypatch.setattr(rocm_isa, "disassemble", lambda payload, chip=None: listing.lower())
    evidence = oracle.selected_symbol_isa_evidence(b"\x7fELF", "gemm")
    core = KI.selected_instruction_stream(listing, "gemm")
    assert evidence["instruction_stream_sha256"] == hashlib.sha256(core.encode()).hexdigest()
    production, count = KI.image_instruction_stream(listing)
    assert production == "<gemm>:\n" + core
    assert count == evidence["instruction_count"]


def test_an_undecodable_word_fails_closed():
    """P1-1: two kernels differing only in an undecodable word. The oracle's
    lenient walk drops the word and hashes them equal; production refuses."""
    word_a = LISTING.replace(
        "\ts_endpgm", "\t.long 0xdeadbeef                  // 000000001AC4: DEADBEEF\n\ts_endpgm")
    word_b = word_a.replace(".long 0xdeadbeef", ".long 0xcafef00d").replace(
        "DEADBEEF", "CAFEF00D")
    lenient = [KI.selected_instruction_stream(t, "gemm") for t in (word_a, word_b)]
    assert lenient[0] == lenient[1], "the lenient walk cannot see the difference"
    for text in (word_a, word_b,
                 LISTING.replace("\ts_endpgm", "\t<unknown>   // 000000001AC4: 0\n\ts_endpgm"),
                 LISTING.replace("\ts_endpgm", "\t\t...\n\ts_endpgm")):
        with pytest.raises(KI.KernelIdentityUnavailable, match="undecod"):
            KI.image_instruction_stream(text)


def _fake_objdump(tmp_path, listing: str, kd: str, version: str = "LLVM version 23.fake") -> str:
    (tmp_path / "text.txt").write_text(listing)
    (tmp_path / "kd.txt").write_text(kd)
    tool = tmp_path / "llvm-objdump"
    tool.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        case "$*" in
          *--version*) echo "LLVM (http://llvm.org/):"; echo "  {version}" ;;
          *-D*) cat "{tmp_path / 'kd.txt'}" ;;
          *) cat "{tmp_path / 'text.txt'}" ;;
        esac
    """))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return str(tool)


def _elf(*, table: bytes = b"", kd: bytes = b"\x11" * 64, trailer: bytes = b"") -> bytes:
    """A minimal ELF64 code object: `.text`, `.rodata` = the 64-byte `gemm.kd`
    descriptor followed by `table`, a symbol table naming `gemm` and `gemm.kd`.
    `trailer` appends non-section bytes (a different build of the same code)."""
    text = b"\x00" * 16
    rodata = kd + table
    shstr = b"\0.text\0.rodata\0.symtab\0.strtab\0.shstrtab\0"
    strtab = b"\0gemm\0gemm.kd\0"
    rodata_addr, text_addr = 0x8C0, 0x1900
    syms = (b"\0" * 24
            + struct.pack("<IBBHQQ", 1, 0x12, 0, 1, text_addr, len(text))
            + struct.pack("<IBBHQQ", 6, 0x11, 0, 2, rodata_addr, 64))
    body = bytearray(b"\0" * 64)
    offsets = {}
    for name, blob in (("text", text), ("rodata", rodata), ("symtab", syms),
                       ("strtab", strtab), ("shstrtab", shstr)):
        offsets[name] = len(body)
        body += blob
    shoff = len(body)
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 0x6, text_addr, offsets["text"], len(text), 0, 0, 256, 0),
        (7, 1, 0x2, rodata_addr, offsets["rodata"], len(rodata), 0, 0, 64, 0),
        (15, 2, 0, 0, offsets["symtab"], len(syms), 4, 1, 8, 24),
        (23, 3, 0, 0, offsets["strtab"], len(strtab), 0, 0, 1, 0),
        (31, 3, 0, 0, offsets["shstrtab"], len(shstr), 0, 0, 1, 0),
    ]
    for sec in sections:
        body += struct.pack("<IIQQQQIIQQ", *sec)
    body[0:4] = b"\x7fELF"
    body[4], body[5], body[6] = 2, 1, 1
    struct.pack_into("<Q", body, 0x28, shoff)
    struct.pack_into("<HHH", body, 0x3A, 64, len(sections), 5)
    return bytes(body) + trailer


ELF = _elf()


def test_constant_data_beyond_the_descriptor_is_digested():
    """P1-2: two kernels reading a constant table that differs in one value
    must not share an identity; the descriptor's own bytes (decoded
    separately, and carrying a layout-dependent code offset) are excluded."""
    base = KI.image_data_digest(_elf(table=b"\x00\x00\x80\x3f"))
    assert base[0] == ".rodata:4"
    assert KI.image_data_digest(_elf(table=b"\x00\x00\x00\x40"))[1] != base[1]
    assert KI.image_data_digest(_elf(table=b"\x00\x00\x80\x3f", kd=b"\x22" * 64)) == base
    assert KI.image_data_digest(_elf(table=b"\x00\x00\x80\x3f", trailer=b"build-2")) == base
    assert KI.image_data_digest(ELF)[0] == ".rodata:0"
    with pytest.raises(KI.KernelIdentityUnavailable, match="malformed|ELF64"):
        KI.image_data_digest(b"\x7fELF" + b"\x00" * 60)


def test_identity_of_an_image(tmp_path):
    tool = _fake_objdump(tmp_path, LISTING, KD)
    identity = KI.hsaco_kernel_identity(ELF, entry_symbol="gemm", isa="gfx1151", objdump=tool)
    assert identity["identity"] == "kernel_code"
    assert identity["normalization"] == KI.NORMALIZATION
    assert identity["instruction_count"] == "7"
    assert identity["instruction_stream_sha256"] == _stream_digest(LISTING)
    assert identity["data_sections"] == ".rodata:0"
    assert identity["disassembler"] == "LLVM version 23.fake"
    # Different image bytes (a different build), same code: same identity.
    other = tmp_path / "other-build"
    other.mkdir()
    rebuilt = _fake_objdump(other, LISTING_REBUILT, KD)
    assert KI.hsaco_kernel_identity(_elf(trailer=b"build-2"), entry_symbol="gemm",
                                    isa="gfx1151", objdump=rebuilt) == identity
    # A different disassembler is a NAMED miss, not an opaque one.
    newer = tmp_path / "newer-tool"
    newer.mkdir()
    changed = KI.hsaco_kernel_identity(
        ELF, entry_symbol="gemm", isa="gfx1151",
        objdump=_fake_objdump(newer, LISTING, KD, version="LLVM version 24.fake"))
    assert KI.identity_mismatch(identity, changed) == ["disassembler"]
    table = KI.hsaco_kernel_identity(_elf(table=b"\x01\x02"), entry_symbol="gemm",
                                     isa="gfx1151", objdump=tool)
    assert KI.identity_mismatch(identity, table) == ["data_sections", "data_sha256"]


@pytest.mark.parametrize("payload, entry, why", [
    (b"not an elf", "gemm", "not an ELF"),
    (b"\x7fELF" + b"\x00" * 60, "gemm", "malformed|ELF64"),
    (ELF, "fa", "not a function"),
])
def test_unidentifiable_images_raise(tmp_path, payload, entry, why):
    tool = _fake_objdump(tmp_path, LISTING, KD)
    with pytest.raises(KI.KernelIdentityUnavailable, match=why):
        KI.hsaco_kernel_identity(payload, entry_symbol=entry, isa="gfx1151", objdump=tool)


def test_a_missing_disassembler_is_a_miss(monkeypatch):
    KI.clear_kernel_identity_cache()
    monkeypatch.setattr(KI, "find_llvm_objdump", lambda: None)
    key = ("cand", "gfx1151", 64, 64, 64, "missing-tool")
    assert KI.compiler_kernel_identity(key, lambda: (ELF, "gemm"), isa="gfx1151") is None
    assert "no llvm-objdump" in KI.miss_reason(key)


def test_a_missing_image_is_a_miss():
    KI.clear_kernel_identity_cache()

    def no_image():
        raise RuntimeError("tessera-opt not built")

    key = ("cand", "no-image")
    assert KI.compiler_kernel_identity(key, no_image, isa="gfx1151") is None
    assert "tessera-opt not built" in KI.miss_reason(key)


def test_the_lookup_disassembles_each_image_once(monkeypatch):
    """Every lookup asks the build path for the image the launch would run
    (its own content-addressed cache), and an unchanged image is served its
    memoized identity without disassembling again (AUTOTUNE-KERNEL-IDENTITY-MEMO:
    the build is no longer skipped, or a changed image would read the old
    identity)."""
    KI.clear_kernel_identity_cache()
    calls = {"build": 0, "disasm": 0}

    def build():
        calls["build"] += 1
        return ELF, "gemm"

    def disasm(payload, *, entry_symbol, isa, objdump=None):
        calls["disasm"] += 1
        return {"identity": "kernel_code", "instruction_stream_sha256": "x"}

    monkeypatch.setattr(KI, "hsaco_kernel_identity", disasm)
    first = KI.compiler_kernel_identity(("c", 1, "opt-A"), build, isa="gfx1151")
    for _ in range(5):
        assert KI.compiler_kernel_identity(("c", 1, "opt-A"), build, isa="gfx1151") == first
    assert calls == {"build": 6, "disasm": 1}
    # A rebuilt compiler is a new key, but the same image bytes are not
    # disassembled twice.
    assert KI.compiler_kernel_identity(("c", 1, "opt-B"), build, isa="gfx1151") == first
    assert calls == {"build": 7, "disasm": 1}
    assert first["generator"] == "tessera-opt"


def test_the_objdump_list_is_shared_with_the_fixture_helper():
    from tests._support import rocm_isa

    assert rocm_isa._candidates() == KI.llvm_objdump_candidates()


# ── the arbiter: served across builds, missed on a changed kernel ───────────

class _Region:
    dtype = "float16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _Generated(Candidate):
    """A compiler-generated Tier-3 candidate whose image is a fake."""
    op = OP_MATMUL
    tier = Tier.HAND_TUNED

    def __init__(self, name, target):
        self.name, self.target = name, target
        self.listing = LISTING
        self.generator = "tessera-opt-build-A"
        self.timed = 0
        self.tool_dir = None

    def artifact_identity(self, region, *inputs):
        if len(inputs) < 2:
            return None
        m, k = inputs[0].shape
        n = inputs[1].shape[1]
        _fake_objdump(self.tool_dir, self.listing, KD)
        key = (self.name, m, n, k, self.generator, hash(self.listing))
        return KI.compiler_kernel_identity(
            key, lambda: (ELF + self.generator.encode() + self.listing.encode(), "gemm"),
            isa="gfx1151")

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), "fake_generated"

    def measure_device_latency(self, region, *inputs, reps=100, warmup=10):
        self.timed += 1
        return 1.0


def test_verdict_is_served_across_builds_and_misses_on_a_changed_kernel(tmp_path, monkeypatch):
    KI.clear_kernel_identity_cache()
    cand = _Generated("kid_generated", "kid_target")
    cand.tool_dir = tmp_path
    monkeypatch.setattr(KI, "find_llvm_objdump", lambda: str(tmp_path / "llvm-objdump"))
    register_candidate(cand)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)).astype(np.float32)
    B = rng.standard_normal((4, 4)).astype(np.float32)
    cache = AT.MeasureCache()

    def race():
        return AT.measured_arbitrate(
            _Region(), OP_MATMUL, "kid_target", A, B, dims=(4, 4, 4), dtype="float16",
            cache=cache, device="fakedev", timing=AT.TIMING_DEVICE, device_repeats=1)

    assert race().name == "kid_generated"
    rec = next(iter(cache._store.values()))
    stamped = rec.evidence["delegate_identities"]["kid_generated"]
    assert stamped["identity"] == "kernel_code"
    timed = cand.timed

    # Another build tree: a different compiler binary (and image bytes) that
    # generates the same kernel. Served from the record, not re-timed.
    cand.generator = "tessera-opt-build-B"
    reloaded = AT.MeasureCache()
    reloaded.load_dict(cache.to_dict())
    cache = reloaded
    race()
    assert cand.timed == timed, "same kernel from another build must be served"
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, "kid_target", A, B, dims=(4, 4, 4), dtype="float16",
        cache=cache, device="fakedev", timing=AT.TIMING_DEVICE) == "kid_generated"

    # The kernel changed: miss, and re-measure.
    cand.listing = LISTING.replace("s_cbranch_vccnz 1027", "s_cbranch_vccnz 1030")
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, "kid_target", A, B, dims=(4, 4, 4), dtype="float16",
        cache=cache, device="fakedev", timing=AT.TIMING_DEVICE) is None
    race()
    assert cand.timed > timed, "a changed kernel must be re-measured"

    # No operands to derive the workload from: the identity cannot be computed,
    # so the verdict is not served (fail closed).
    assert AT.corpus_winner(
        _Region(), OP_MATMUL, "kid_target", dims=(4, 4, 4), dtype="float16",
        cache=cache, device="fakedev", timing=AT.TIMING_DEVICE) is None


def test_every_tier_requires_an_identity_and_none_cannot_opt_out():
    """AUTOTUNE-EMITTED-IDENTITY-2026-09-27: an EMITTED candidate with no
    identity used to be served on the pin alone. It now misses, and an
    override of `requires_artifact_identity` returning False does not change
    that -- the arbiter no longer asks."""
    class _Emitted(Candidate):
        name, target, op, tier = "kid_emitted", "kid_t2", OP_MATMUL, Tier.EMITTED

        def run(self, region, *inputs, **kwargs):
            return None, "x"

    class _OptOut(_Emitted):
        def requires_artifact_identity(self):
            return False

    assert _Emitted().requires_artifact_identity()
    rec = AT.MeasureRecord(winner="kid_emitted", latency_ms=1.0,
                           candidates={"kid_emitted": 1.0}, unmeasured={})
    assert not AT._record_matches_live_delegates(rec, {"kid_emitted": _Emitted()})
    assert not AT._record_matches_live_delegates(rec, {"kid_emitted": _OptOut()})


def test_rocm_candidates_derive_their_key_from_the_workload(monkeypatch, tmp_path):
    """`rocm_wmma_gemm` identifies the image `_rocm_wmma_fused_2d` would launch
    for (M, N, K, epilogue); `rocm_flash_attn` the FA-2 image at Q's head_dim.
    Without operands there is no workload, so no identity."""
    from tessera import runtime as rt
    from tessera.compiler import fusion as F
    from tessera.compiler.emit import rocm_hip

    KI.clear_kernel_identity_cache()
    built: list[tuple] = []

    def fused_image(m, n, k, dtype, *, bias, activation, **_):
        built.append((m, n, k, dtype, bias, activation))
        return _elf(trailer=f"{m}x{n}x{k}".encode()), "gemm"

    monkeypatch.setattr(rt, "_rocm_wmma_fused_image", fused_image)
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1151")
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1151")
    monkeypatch.setattr(KI, "find_llvm_objdump", lambda: _fake_objdump(tmp_path, LISTING, KD))
    monkeypatch.setattr(KI, "generator_fingerprint", lambda: "sha256:opt")
    wmma = rocm_hip.RocmWmmaGemmCandidate.__new__(rocm_hip.RocmWmmaGemmCandidate)
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    a = np.zeros((64, 32), np.float32)
    b = np.zeros((32, 48), np.float32)
    identity = wmma.artifact_identity(region, a, b, np.zeros(48, np.float32))
    assert identity is not None and identity["entry"] == "gemm"
    assert built == [(64, 48, 32, "f16", True, "gelu")]
    assert wmma.artifact_identity(region, a, b) == identity
    # Each lookup asks the launch's own build path for the image (which is
    # where an in-process generator change would show up); the identity of an
    # unchanged image is served from the memo.
    assert built == [(64, 48, 32, "f16", True, "gelu")] * 2
    assert wmma.artifact_identity(region) is None   # no workload
    assert wmma.artifact_identity(F.FusedRegion(epilogue=("gelu", "bias")), a, b) is None
    assert wmma.delegate_identity() is None          # no binary-digest key any more
    # The ISA is the live device's, and only when it is the build target.
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: None)
    assert wmma.artifact_identity(region, a, b) is None
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1201")
    assert wmma.artifact_identity(region, a, b) is None
    monkeypatch.setattr(rt, "_rocm_device_name", lambda: "gfx1151")

    fa_built: list[int] = []

    def fa_image(head_dim, dtype="f16", **_):
        fa_built.append(head_dim)
        return _elf(trailer=b"fa").replace(b"gemm", b"fa\0\0"), "fa"

    monkeypatch.setattr(rt, "_rocm_flash_attn_image", fa_image)
    monkeypatch.setattr(KI, "find_llvm_objdump",
                        lambda: _fake_objdump(tmp_path, LISTING.replace("<gemm>", "<fa>"),
                                              KD.replace("gemm", "fa")))

    class _Attn:
        def _natural(self, Q, K):
            return np.asarray(Q), np.asarray(K)

    fa = rocm_hip.RocmFlashAttnCandidate()
    q = np.zeros((16, 64), np.float32)
    assert fa.artifact_identity(_Attn(), q, q, q)["entry"] == "fa"
    assert fa_built == [64]
    assert fa.artifact_identity(_Attn(), q, q) is None


def test_measured_arbitrate_keys_on_the_dims_production_infers(tmp_path, monkeypatch):
    """P1-3: a recorder that passes no dims must land under the (M, N, K)
    bucket `corpus_winner` infers for ordinary dispatch."""
    KI.clear_kernel_identity_cache()
    cand = _Generated("kid_dims", "kid_dims_target")
    cand.tool_dir = tmp_path
    monkeypatch.setattr(KI, "find_llvm_objdump", lambda: str(tmp_path / "llvm-objdump"))
    register_candidate(cand)
    A = np.ones((8, 16), np.float32)
    B = np.ones((16, 4), np.float32)
    cache = AT.MeasureCache()
    AT.measured_arbitrate(_Region(), OP_MATMUL, "kid_dims_target", A, B, dtype="float16",
                          cache=cache, device="fakedev", timing=AT.TIMING_DEVICE,
                          device_repeats=1)
    (key,) = cache._store
    assert key[3] == AT.bucket_key((8, 4, 16), AT.SpecPolicy.BUCKET)
    assert AT.corpus_winner(_Region(), OP_MATMUL, "kid_dims_target", A, B,
                            cache=cache, device="fakedev",
                            timing=AT.TIMING_DEVICE) == "kid_dims"


def test_committed_rocm_fused_rows_carry_kernel_code_identities():
    """The gfx1151 fused_region rows are stamped with the v2 kernel-code
    identity and keyed on the (M, N, K) bucket ordinary dispatch infers
    (re-recorded 2026-09-26 on Princess-Luna)."""
    import json

    rows = [r for r in json.loads(AT.corpus_path().read_text())["records"]
            if r["device"] == "rocm:gfx1151" and r["op"] == "fused_region"]
    assert len(rows) == 8
    for row in rows:
        assert len(row["bucket"]) == 3, "a 2-D bucket is never looked up by dispatch"
        identity = row["evidence"]["delegate_identities"]["rocm_wmma_gemm"]
        assert identity["identity"] == "kernel_code"
        assert identity["normalization"] == KI.NORMALIZATION
        assert identity["isa"] == "gfx1151" and identity["entry"] == "gemm"
        assert identity["data_sections"] == ".rodata:0"
        assert identity["disassembler"]
        assert "abi_digest" not in identity
