"""Identity of a compiler-generated kernel by the code it executes (Decision #11).

A measured autotune verdict is valid only for the code that was timed. For a
kernel ``tessera-opt`` generates at run time, the first artifact identity keyed
the verdict on the digest of the ``tessera-opt`` *binary*. That is sound in the
safe direction (a rebuilt compiler always misses) but useless in the other: the
binary's bytes differ between any two builds, even of one commit in two build
directories, so a committed row was served only in the tree that recorded it.

The identity here is the **instruction stream of the image the candidate would
run** for one workload key, so two builds that generate the same kernel share
it and a change to the kernel changes it. It is a digest of *decoded*
instructions, not of image bytes, because image bytes carry build-specific
noise: gfx1201 folded/packed MXFP4 HSACOs differ as whole payloads between two
builds of one revision while their instruction streams are identical
(``benchmarks/baselines/gfx1201_mxfp4_producer_relabel_20260924``).

Normalization ``tessera.kernel_code.v2`` (AMDGPU HSACO)
-------------------------------------------------------

``llvm-objdump -d`` of the image, lower-cased. The per-instruction rule is the
one ``benchmarks/rocm/inspect_gfx1201_folded_prefill.selected_symbol_isa_evidence``
applies; that file is frozen by the sealed gfx1201 packets that bind its hash,
so it is kept as a declared oracle (Decision #31) and
``tests/unit/test_kernel_code_identity.py`` checks that the shared
per-instruction core (:func:`instruction_blocks`) reproduces its digest and
that the production stream is that core's output plus function headers. The
oracle drops undecodable words; production refuses them (below), so the two
agree only on fully decoded listings -- the only kind production accepts.

* **Kept:** every function symbol in ``.text``, in image order, by name; and
  each instruction line's text *before* the ``//`` comment with whitespace
  collapsed. Branch operands are PC-relative word offsets and stay; literal
  constants stay.
* **Refused (fail closed):** any other non-blank line inside a function -- a
  ``.long``/``.short`` word the disassembler could not decode, ``<unknown>``,
  a ``...`` elision. Dropping them would let two kernels that differ only in
  those words hash equal (the realistic case: a disassembler older than the
  compiler printing a new gfx12 WMMA/SWMMAC form as ``.long``).
* **Dropped:** the ``//`` comment on every instruction line, which holds the
  instruction's address, its raw encoding and the ``<sym+0x..>`` branch-target
  annotation; symbol addresses on the ``<sym>:`` headers; the file header
  (it names the temporary file); loader and metadata sections -- ``.note``
  (AMDGPU metadata), ``.comment`` (linker version string), ``.dynamic``,
  ``.dynsym``, ``.dynstr``, ``.hash``/``.gnu.hash``, ``.relro_padding``,
  ``.symtab``/``.strtab``.
* **Kept separately:** the entry kernel's descriptor (``<entry>.kd``, decoded
  by ``llvm-objdump -D --disassemble-symbols=<entry>.kd``): the
  ``.amdhsa_kernel`` block, whitespace-collapsed. It fixes VGPR/SGPR
  allocation, LDS and scratch size and the float modes -- what the hardware
  runs the stream *with*.
* **Data:** every other allocatable, non-executable section (``.rodata``
  constant tables, ``.data``, ``.bss`` size) is digested byte for byte with the
  kernel descriptors' byte ranges removed (they are covered, decoded, above,
  and hold a layout-dependent code offset). ``data_sections`` names each such
  section and how many bytes were digested, so "the image carries no constant
  data" is readable from committed evidence as ``.rodata:0``.
* **Disassembler:** the ``llvm-objdump --version`` line, so a tool change reads
  as a named field mismatch rather than an opaque digest change. The tool's
  path is not in the identity (the same tool sits at different paths on
  different boxes); recorders print it.

Limits, stated so they are not read as covered: the host-side launch geometry
is not part of the image (it derives from the same schedule that selects the
image, and the workload key); data holding absolute addresses would differ
across layouts (a false miss, never a false hit); and the digest is of what
the disassembler decodes, so a decoder that printed two different encodings
identically would conflate them.

**Fail closed.** No disassembler, an image that will not disassemble or parse,
an undecodable word, a missing entry symbol or descriptor, or an empty stream
is ``None`` -- a miss, never a guess. :func:`compiler_kernel_identity` records
why in :func:`miss_reason`; :func:`identity_mismatch` names the fields that
differ between a recorded and a live identity.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Hashable

#: Normalization version. Bump it when a rule above changes: every stamped row
#: then misses, which is the correct outcome for a changed definition.
NORMALIZATION = "tessera.kernel_code.v2"

_HEADER = re.compile(r"[0-9a-f]+ <([^>]+)>:")
_MNEMONIC = re.compile(r"^[a-z][a-z0-9_]+\b")


class KernelIdentityUnavailable(RuntimeError):
    """The identity of an image could not be established (fail closed)."""


def llvm_objdump_candidates() -> list[Path]:
    """Where an LLVM ``llvm-objdump`` that decodes AMDGPU lives on the fleet,
    most explicit first. ``tests/_support/rocm_isa`` resolves through this list
    too, so the fixture helper and the identity cannot disagree on the tool."""
    found: list[Path] = []
    explicit = os.environ.get("TESSERA_LLVM_BIN")
    if explicit:
        found.append(Path(explicit) / "llvm-objdump")
    rocm = os.environ.get("ROCM_PATH")
    if rocm:
        found.append(Path(rocm) / "llvm" / "bin" / "llvm-objdump")
    found += [
        Path.home() / ".local/share/tessera-toolchains/llvm-23.1.1/bin/llvm-objdump",
        Path("/opt/rocm/core/llvm/bin/llvm-objdump"),
        Path("/opt/rocm/llvm/bin/llvm-objdump"),
        Path("/usr/lib/llvm-23/bin/llvm-objdump"),
    ]
    return found


def find_llvm_objdump() -> str | None:
    """The disassembler, or ``None`` when no fleet location (nor ``PATH``) has one."""
    for candidate in llvm_objdump_candidates():
        if candidate.is_file():
            return str(candidate)
    return shutil.which("llvm-objdump")


def instruction_blocks(disassembly: str, *, strict: bool = False
                       ) -> list[tuple[str, list[str]]]:
    """``[(symbol, [normalized instruction, ...]), ...]`` for every ``<sym>:``
    block of an ``llvm-objdump -d`` listing, in listing order (rules above).

    ``strict=False`` is the oracle's behaviour: a line that is not a decoded
    instruction is skipped. ``strict=True`` (production) raises
    :class:`KernelIdentityUnavailable` on any non-blank line inside a function
    that is not a decoded instruction -- an undecodable ``.long`` word,
    ``<unknown>``, a ``...`` elision."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in disassembly.lower().splitlines():
        header = _HEADER.fullmatch(line.strip())
        if header:
            current = []
            blocks.append((header.group(1), current))
            continue
        if current is None:
            continue
        if "//" not in line:
            if strict and line.strip():
                raise KernelIdentityUnavailable(
                    f"undecoded line in {blocks[-1][0]}: {line.strip()[:80]!r}")
            continue
        instruction = " ".join(line.split("//", 1)[0].split())
        if _MNEMONIC.match(instruction) and "<unknown>" not in instruction:
            current.append(instruction)
        elif strict:
            raise KernelIdentityUnavailable(
                f"undecodable word in {blocks[-1][0]}: {instruction[:80]!r}")
    return blocks


def selected_instruction_stream(disassembly: str, entry_symbol: str) -> str:
    """The normalized stream of exactly one symbol, newline-terminated: the text
    whose sha256 the gfx1201 packets record as ``instruction_stream_sha256``.
    Raises :class:`KernelIdentityUnavailable` unless the symbol appears once."""
    matches = [instrs for name, instrs in instruction_blocks(disassembly)
               if name == entry_symbol.lower()]
    if len(matches) != 1:
        raise KernelIdentityUnavailable(
            f"image must contain exactly one {entry_symbol} symbol; found {len(matches)}")
    return "\n".join(matches[0]) + "\n"


def image_instruction_stream(disassembly: str) -> tuple[str, int]:
    """Every function block, as ``<symbol>:`` then its instructions: the whole
    image's normalized stream, plus its instruction count. Strict: an
    undecodable word raises (see :func:`instruction_blocks`)."""
    lines: list[str] = []
    count = 0
    for name, instrs in instruction_blocks(disassembly, strict=True):
        lines.append(f"<{name}>:")
        lines.extend(instrs)
        count += len(instrs)
    return "\n".join(lines) + "\n", count


#: Allocatable sections that are loader/metadata, not data a kernel reads.
_LOADER_SECTIONS = frozenset({
    ".note", ".dynsym", ".dynstr", ".hash", ".gnu.hash", ".dynamic", ".relro_padding",
})


def image_data_digest(payload: bytes) -> tuple[str, str]:
    """``(data_sections, data_sha256)`` of every allocatable, non-executable,
    non-loader section of an ELF64 code object, with ``*.kd`` symbol byte
    ranges removed. ``data_sections`` is ``"name:bytes,..."`` (``"none"`` when
    the image has no such section). Raises on a malformed ELF."""
    try:
        if payload[4] != 2 or payload[5] != 1:
            raise KernelIdentityUnavailable("image is not a little-endian ELF64 object")
        shoff, = struct.unpack_from("<Q", payload, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", payload, 0x3A)
        headers = [struct.unpack_from("<IIQQQQIIQQ", payload, shoff + i * shentsize)
                   for i in range(shnum)]
        names_hdr = headers[shstrndx]
        names = payload[names_hdr[4]:names_hdr[4] + names_hdr[5]]

        def name_of(off: int, table: bytes) -> str:
            return table[off:table.index(b"\0", off)].decode()

        kd_ranges: dict[int, list[tuple[int, int]]] = {}
        for _n, sh_type, _f, _a, sh_off, sh_size, sh_link, *_ in headers:
            if sh_type != 2:                      # SHT_SYMTAB
                continue
            strtab = headers[sh_link]
            strings = payload[strtab[4]:strtab[4] + strtab[5]]
            for i in range(sh_size // 24):
                st_name, _info, _other, st_shndx, st_value, st_size = struct.unpack_from(
                    "<IBBHQQ", payload, sh_off + i * 24)
                if st_name and 0 < st_shndx < shnum and name_of(st_name, strings).endswith(".kd"):
                    base = headers[st_shndx][3]
                    kd_ranges.setdefault(st_shndx, []).append(
                        (st_value - base, st_value - base + st_size))
        described: list[str] = []
        digest = hashlib.sha256()
        for index, (sh_name, sh_type, sh_flags, _a, sh_off, sh_size, *_r) in enumerate(headers):
            name = name_of(sh_name, names)
            if not sh_flags & 0x2 or sh_flags & 0x4 or name in _LOADER_SECTIONS or sh_type == 7:
                continue                          # not ALLOC, EXECINSTR, loader, NOTE
            if sh_type == 8:                      # NOBITS: only its size exists
                described.append(f"{name}:nobits{sh_size}")
                digest.update(f"{name}\0nobits{sh_size}\0".encode())
                continue
            data = bytearray(payload[sh_off:sh_off + sh_size])
            for lo, hi in sorted(kd_ranges.get(index, []), reverse=True):
                del data[max(lo, 0):max(hi, 0)]
            described.append(f"{name}:{len(data)}")
            digest.update(name.encode() + b"\0" + bytes(data) + b"\0")
    except KernelIdentityUnavailable:
        raise
    except (struct.error, IndexError, ValueError, UnicodeDecodeError) as exc:
        raise KernelIdentityUnavailable(f"malformed ELF code object: {exc}") from exc
    return (",".join(described) or "none"), digest.hexdigest()


_TOOL_VERSIONS: dict[str, str] = {}


def disassembler_version(objdump: str) -> str:
    """The ``llvm-objdump --version`` line naming the LLVM version (cached)."""
    cached = _TOOL_VERSIONS.get(objdump)
    if cached is None:
        done = subprocess.run([objdump, "--version"], capture_output=True, text=True)
        lines = [ln.strip() for ln in done.stdout.splitlines() if "version" in ln.lower()]
        if done.returncode != 0 or not lines:
            raise KernelIdentityUnavailable(f"{objdump} --version did not name a version")
        cached = _TOOL_VERSIONS[objdump] = lines[0]
    return cached


def identity_mismatch(recorded: dict[str, str] | None,
                      live: dict[str, str] | None) -> list[str]:
    """The fields on which a recorded and a live identity differ (``[]`` when
    they match): makes a miss readable -- ``["disassembler"]`` for a tool
    change, ``["instruction_stream_sha256", ...]`` for a changed kernel."""
    if recorded is None or live is None:
        return [] if recorded == live else ["<absent>"]
    return sorted(k for k in set(recorded) | set(live) if recorded.get(k) != live.get(k))


def kernel_descriptor_block(listing: str, entry_symbol: str) -> str:
    """The decoded ``.amdhsa_kernel <entry>`` block of ``<entry>.kd``,
    whitespace-collapsed and newline-terminated."""
    lines: list[str] = []
    inside = False
    for line in listing.lower().splitlines():
        text = " ".join(line.split())
        if text == f".amdhsa_kernel {entry_symbol.lower()}":
            inside = True
        if inside and text:
            lines.append(text)
        if inside and text == ".end_amdhsa_kernel":
            return "\n".join(lines) + "\n"
    raise KernelIdentityUnavailable(f"no decoded kernel descriptor for {entry_symbol}.kd")


def _run_objdump(objdump: str, args: list[str], payload: bytes) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False)
    try:
        handle.write(payload)
        handle.close()
        done = subprocess.run([objdump, *args, handle.name], capture_output=True, text=True)
    finally:
        os.unlink(handle.name)
    if done.returncode != 0 or not done.stdout.strip():
        raise KernelIdentityUnavailable(
            f"llvm-objdump {' '.join(args)} failed: rc={done.returncode} {done.stderr.strip()[:300]}")
    return done.stdout


def hsaco_kernel_identity(payload: bytes, *, entry_symbol: str, isa: str,
                          objdump: str | None = None) -> dict[str, str]:
    """The ``tessera.kernel_code.v1`` identity of one AMDGPU code object.

    Raises :class:`KernelIdentityUnavailable` when it cannot be established."""
    if not payload or payload[:4] != b"\x7fELF":
        raise KernelIdentityUnavailable("image is not an ELF code object")
    tool = objdump or find_llvm_objdump()
    if tool is None:
        raise KernelIdentityUnavailable(
            "no llvm-objdump (set TESSERA_LLVM_BIN or source scripts/_rocm_env.sh)")
    mcpu = [f"--mcpu={isa}"] if isa else []
    data_sections, data_sha256 = image_data_digest(payload)
    stream, count = image_instruction_stream(_run_objdump(tool, ["-d", *mcpu], payload))
    if count == 0:
        raise KernelIdentityUnavailable("image disassembled to no instructions")
    if f"<{entry_symbol.lower()}>:" not in stream.splitlines():
        raise KernelIdentityUnavailable(f"entry symbol {entry_symbol} is not a function of the image")
    descriptor = kernel_descriptor_block(
        _run_objdump(tool, ["-D", f"--disassemble-symbols={entry_symbol}.kd", *mcpu], payload),
        entry_symbol)
    return {
        "identity": "kernel_code",
        "normalization": NORMALIZATION,
        "isa": isa,
        "entry": entry_symbol,
        "instruction_stream_sha256": hashlib.sha256(stream.encode()).hexdigest(),
        "kernel_descriptor_sha256": hashlib.sha256(descriptor.encode()).hexdigest(),
        "instruction_count": str(count),
        "data_sections": data_sections,
        "data_sha256": data_sha256,
        "disassembler": disassembler_version(tool),
    }


#: Per-process identity memo: caller key -> the image it was computed from and
#: its identity, or ``None`` for a miss. The image is kept so a lookup can
#: prove the launch would still run *that* image (below).
_IDENTITIES: dict[Hashable, tuple[bytes, str, dict[str, str]] | None] = {}
#: payload sha256 -> identity, so two keys that build one image disassemble once.
_BY_PAYLOAD: dict[tuple[str, str, str, str], dict[str, str]] = {}
_MISS_REASONS: dict[Hashable, str] = {}


def compiler_kernel_identity(
    key: Hashable,
    build_image: Callable[[], tuple[bytes, str]],
    *,
    isa: str,
    generator: str = "tessera-opt",
) -> dict[str, str] | None:
    """The kernel-code identity of the image ``build_image`` returns now.

    ``build_image`` returns ``(payload, entry_symbol)`` through the candidate's
    own build path -- the statement of the selection its launch goes through,
    and so its content-addressed image cache (the hsaco caches key on the
    directive text the Python generator produced). **It is called on every
    lookup**, and the identity is of the image it returns: the memo under
    ``key`` is reused only when that image is byte-identical to the one the
    memo was computed from. ``key`` still names the candidate, the workload's
    kernel-selecting facts and the generator binary, but it no longer has to
    name everything that selects the image -- an in-process change to the
    Python directive generator builds a new image, and a new image is
    re-identified rather than served the old digest
    (``AUTOTUNE-KERNEL-IDENTITY-MEMO``, closed 2026-09-27). The per-lookup cost
    is the launch's own cache lookup plus a byte compare; the disassembler runs
    once per distinct image.

    Any failure is a ``None``: a miss, with :func:`miss_reason`. A key whose
    build or identification failed stays a miss for the process (a cached
    ``None`` can never serve a verdict, so it cannot be a false hit)."""
    if key in _IDENTITIES and _IDENTITIES[key] is None:
        return None
    try:
        payload, entry = build_image()
        payload = bytes(payload)
        memo = _IDENTITIES.get(key)
        if memo is not None and memo[1] == entry and (
                memo[0] is payload or memo[0] == payload):
            return dict(memo[2])
        digest_key = (hashlib.sha256(payload).hexdigest(), entry, isa,
                      find_llvm_objdump() or "")
        identity = _BY_PAYLOAD.get(digest_key)
        if identity is None:
            identity = hsaco_kernel_identity(payload, entry_symbol=entry, isa=isa)
            _BY_PAYLOAD[digest_key] = identity
        result = {"generator": generator, **identity}
    except Exception as exc:  # noqa: BLE001 - any failure to identify is a miss
        _MISS_REASONS[key] = f"{type(exc).__name__}: {exc}"
        _IDENTITIES[key] = None
        return None
    _MISS_REASONS.pop(key, None)
    _IDENTITIES[key] = (payload, entry, dict(result))
    return result


def miss_reason(key: Hashable) -> str | None:
    """Why :func:`compiler_kernel_identity` returned ``None`` for ``key``."""
    return _MISS_REASONS.get(key)


def clear_kernel_identity_cache() -> None:
    _IDENTITIES.clear()
    _BY_PAYLOAD.clear()
    _MISS_REASONS.clear()
    _TOOL_VERSIONS.clear()


def cache_size() -> int:
    return len(_IDENTITIES)


def generator_fingerprint() -> Any:
    """The ``tessera-opt`` identity to fold into a :func:`compiler_kernel_identity`
    key (content digest, cached on the file's mtime/size), or ``None``."""
    from .toolchain_identity import tessera_opt_identity

    identity = tessera_opt_identity()
    return None if identity is None else identity.get("abi_digest")


__all__ = [
    "NORMALIZATION",
    "KernelIdentityUnavailable",
    "cache_size",
    "clear_kernel_identity_cache",
    "compiler_kernel_identity",
    "disassembler_version",
    "find_llvm_objdump",
    "generator_fingerprint",
    "hsaco_kernel_identity",
    "identity_mismatch",
    "image_data_digest",
    "image_instruction_stream",
    "instruction_blocks",
    "kernel_descriptor_block",
    "llvm_objdump_candidates",
    "miss_reason",
    "selected_instruction_stream",
]
