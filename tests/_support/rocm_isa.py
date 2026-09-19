"""One way to ask what instruction a ROCm image actually contains.

A device row that proves a matrix form *executes* is not the same claim as one
that proves the compiler *selected that form*, and the numbers often cannot
tell them apart:

  * RDNA4's double-K int4 (`V_WMMA_I32_16X16X32_IU4`) and two K=16 int4 WMMAs
    compute the identical exact integer product, so an exactness assertion
    passes either way -- the ISA is the only witness that the new K shape was
    emitted at all.
  * A mixed OCP FP8 pair that silently fell back to `FP8_FP8` reads one operand
    in the wrong format, and with fp8's narrow range that error looks like
    ordinary accumulation noise.

So the mnemonic is load-bearing evidence, and before this module each site
carried its own copy of finding `llvm-objdump` -- with three different policies
for not finding it, one of which (`if objdump is not None:`) dropped the
instruction claim and still reported a pass. That is the hollow-green shape:
the name said `select_their_instruction` and nothing had been selected.

**A missing disassembler here is a failure, not a skip.** These callers have
already asserted they are the owning device (`TESSERA_GFX1201_DEVICE_PROOF=1`
plus a live-arch check), and every ROCm installation ships `llvm-objdump` under
`$ROCM_PATH/llvm/bin`, so absence means a broken fixture environment rather
than an honest capability gap. Skipping would be the more comfortable policy
and the less honest one: it converts "this was not evaluated" into a green row.
"""

from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

#: Where an LLVM 23 `llvm-objdump` lives on the fleet, most explicit first.
def _candidates() -> list[Path]:
    found: list[Path] = []
    explicit = os.environ.get("TESSERA_LLVM_BIN")
    if explicit:
        found.append(Path(explicit) / "llvm-objdump")
    rocm = os.environ.get("ROCM_PATH")
    if rocm:
        found.append(Path(rocm) / "llvm" / "bin" / "llvm-objdump")
    found += [
        Path.home() / ".local/share/tessera-toolchains/llvm-23.1.1/bin/llvm-objdump",
        Path("/opt/rocm/llvm/bin/llvm-objdump"),
        Path("/usr/lib/llvm-23/bin/llvm-objdump"),
    ]
    return found


def llvm_objdump() -> str:
    """The disassembler, or a failure naming every path that was tried."""
    for candidate in _candidates():
        if candidate.is_file():
            return str(candidate)
    on_path = shutil.which("llvm-objdump")
    if on_path:
        return on_path
    raise AssertionError(
        "llvm-objdump is required to assert which instruction was emitted, and "
        "no fleet location has it: "
        + ", ".join(str(c) for c in _candidates())
        + ". Set TESSERA_LLVM_BIN, or source scripts/_rocm_env.sh so ROCM_PATH "
        "resolves the toolkit's own LLVM. This is a failure and not a skip "
        "because the caller has already asserted it is the owning device."
    )


def disassemble(payload: bytes, *, chip: str | None = None) -> str:
    """Disassemble a code object. `chip` is passed as `--mcpu` when given."""
    objdump = llvm_objdump()
    handle = tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False)
    try:
        handle.write(payload)
        handle.close()
        command = [objdump, "-d"] + ([f"--mcpu={chip}"] if chip else []) + [handle.name]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0 or not done.stdout.strip():
            raise AssertionError(
                f"llvm-objdump could not disassemble the image: rc={done.returncode} "
                f"{done.stderr.strip()[:400]}"
            )
        return done.stdout.lower()
    finally:
        os.unlink(handle.name)


def mnemonics(text: str, pattern: str) -> collections.Counter:
    """Histogram of the mnemonics in `text` matching `pattern`."""
    return collections.Counter(re.findall(pattern, text))


def assert_selected(
    payload: bytes,
    *,
    chip: str,
    pattern: str,
    require: str | tuple[str, ...] = (),
    forbid: str | tuple[str, ...] = (),
    what: str = "",
) -> collections.Counter:
    """Assert which mnemonics the image does and does not contain.

    `forbid` is not decoration. A `require`-only assertion cannot see a
    *swap*: for the mixed FP8 pairs, finding `fp8_bf8` does not establish that
    `bf8_fp8` is absent, and an operand exchange somewhere in the stack would
    emit the mirror while still satisfying a presence check. The negative half
    is what makes the positive one mean "this and not the other".
    """
    require = (require,) if isinstance(require, str) else tuple(require)
    forbid = (forbid,) if isinstance(forbid, str) else tuple(forbid)
    text = disassemble(payload, chip=chip)
    seen = mnemonics(text, pattern)
    label = f"{what}: " if what else ""
    missing = [name for name in require if name not in seen]
    assert not missing, (
        f"{label}expected {', '.join(missing)} in the emitted ISA; saw {dict(seen)}"
    )
    present = [name for name in forbid if name in seen]
    assert not present, (
        f"{label}{', '.join(present)} must not be emitted; saw {dict(seen)}"
    )
    return seen
