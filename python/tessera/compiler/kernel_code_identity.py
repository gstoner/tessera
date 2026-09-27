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

Normalization ``tessera.kernel_code.v1`` (AMDGPU HSACO)
-------------------------------------------------------

``llvm-objdump -d`` of the image, lower-cased (the rule
``benchmarks/rocm/inspect_gfx1201_folded_prefill.selected_symbol_isa_evidence``
applies; that file is frozen by the sealed gfx1201 packets that bind its hash,
so it is kept as a declared oracle and ``tests/unit/test_kernel_code_identity.py``
proves the two agree -- Decision #31):

* **Kept:** every function symbol in ``.text``, in image order, by name; and
  each instruction line's text *before* the ``//`` comment with whitespace
  collapsed, when it starts with a mnemonic. Branch operands are PC-relative
  word offsets and stay; literal constants stay.
* **Dropped:** the ``//`` comment on every instruction line, which holds the
  instruction's address, its raw encoding and the ``<sym+0x..>`` branch-target
  annotation; symbol addresses on the ``<sym>:`` headers; the file header
  (it names the temporary file); and every non-code section -- ``.note``
  (AMDGPU metadata), ``.comment`` (linker version string), ``.dynamic``,
  ``.dynsym``, ``.hash``/``.gnu.hash``, ``.symtab``/``.strtab``.
* **Kept separately:** the entry kernel's descriptor (``<entry>.kd``, decoded
  by ``llvm-objdump -D --disassemble-symbols=<entry>.kd``): the
  ``.amdhsa_kernel`` block, whitespace-collapsed. It is not an instruction but
  it fixes VGPR/SGPR allocation, LDS and scratch size and the float modes --
  what the hardware runs the stream *with*, and occupancy is part of what was
  timed.

Limits, stated so they are not read as covered: data sections other than the
kernel descriptor (a ``.rodata`` constant table) are not digested -- the images
in scope have none (their ``.rodata`` is the 64-byte descriptor alone); the
host-side launch geometry is not part of the image (it derives from the same
schedule that selects the image, and the workload key); and the digest is of
what the disassembler decodes, so a decoder that printed two different
encodings identically would conflate them.

**Fail closed.** No disassembler, an image that will not disassemble, a missing
entry symbol or descriptor, or an empty stream is ``None`` -- a miss, never a
guess. :func:`compiler_kernel_identity` records why in :func:`miss_reason`.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Hashable

#: Normalization version. Bump it when a rule above changes: every stamped row
#: then misses, which is the correct outcome for a changed definition.
NORMALIZATION = "tessera.kernel_code.v1"

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


def instruction_blocks(disassembly: str) -> list[tuple[str, list[str]]]:
    """``[(symbol, [normalized instruction, ...]), ...]`` for every ``<sym>:``
    block of an ``llvm-objdump -d`` listing, in listing order (rules above)."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in disassembly.lower().splitlines():
        header = _HEADER.fullmatch(line.strip())
        if header:
            current = []
            blocks.append((header.group(1), current))
            continue
        if current is None or "//" not in line:
            continue
        instruction = " ".join(line.split("//", 1)[0].split())
        if _MNEMONIC.match(instruction):
            current.append(instruction)
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
    image's normalized stream, plus its instruction count."""
    lines: list[str] = []
    count = 0
    for name, instrs in instruction_blocks(disassembly):
        lines.append(f"<{name}>:")
        lines.extend(instrs)
        count += len(instrs)
    return "\n".join(lines) + "\n", count


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
    }


#: Per-process identity cache: caller key -> identity (or ``None`` for a miss).
_IDENTITIES: dict[Hashable, dict[str, str] | None] = {}
#: payload sha256 -> identity, so two keys that build one image disassemble once.
_BY_PAYLOAD: dict[tuple[str, str, str], dict[str, str]] = {}
_MISS_REASONS: dict[Hashable, str] = {}


def compiler_kernel_identity(
    key: Hashable,
    build_image: Callable[[], tuple[bytes, str]],
    *,
    isa: str,
    generator: str = "tessera-opt",
) -> dict[str, str] | None:
    """The kernel-code identity for one compiler-generated candidate and key,
    cached for the life of the process.

    ``key`` must name everything that selects the image: the candidate, the
    workload's kernel-selecting facts (shape, dtype, epilogue, chip) and the
    generator binary's identity -- a rebuilt ``tessera-opt`` is a new key, so it
    is re-identified rather than served a cached digest. ``build_image`` returns
    ``(payload, entry_symbol)`` through the candidate's own build path (and so
    through its content-addressed compile cache); it is called at most once per
    key. Any failure is a cached ``None``: a miss, with :func:`miss_reason`."""
    if key in _IDENTITIES:
        cached = _IDENTITIES[key]
        return None if cached is None else dict(cached)
    try:
        payload, entry = build_image()
        digest_key = (hashlib.sha256(payload).hexdigest(), entry, isa)
        identity = _BY_PAYLOAD.get(digest_key)
        if identity is None:
            identity = hsaco_kernel_identity(payload, entry_symbol=entry, isa=isa)
            _BY_PAYLOAD[digest_key] = identity
        result: dict[str, str] | None = {"generator": generator, **identity}
    except Exception as exc:  # noqa: BLE001 - any failure to identify is a miss
        _MISS_REASONS[key] = f"{type(exc).__name__}: {exc}"
        result = None
    _IDENTITIES[key] = None if result is None else dict(result)
    return result


def miss_reason(key: Hashable) -> str | None:
    """Why :func:`compiler_kernel_identity` returned ``None`` for ``key``."""
    return _MISS_REASONS.get(key)


def clear_kernel_identity_cache() -> None:
    _IDENTITIES.clear()
    _BY_PAYLOAD.clear()
    _MISS_REASONS.clear()


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
    "find_llvm_objdump",
    "generator_fingerprint",
    "hsaco_kernel_identity",
    "image_instruction_stream",
    "instruction_blocks",
    "kernel_descriptor_block",
    "llvm_objdump_candidates",
    "miss_reason",
    "selected_instruction_stream",
]
