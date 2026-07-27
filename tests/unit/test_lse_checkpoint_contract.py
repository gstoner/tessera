"""Guard the `tessera_attn.lse.save` trait against the FA-2 implementation.

`LseSaveOp` is `Pure` only because it is currently degenerate: it takes no
destination, `lse.load` takes no operands, nothing links a load to a save, and
its single emission site discards the result. Marking it Pure stopped a no-op
from behaving like a side effect and unblocked APPLE-ATTN-STREAM-1.

That marking is a trap for whoever implements the real FlashAttention-2
checkpoint. A store must be non-`Pure` and carry `MemWrite`. If a destination
operand is added while `Pure` remains, DCE deletes the store and the backward
reads uninitialized LSE — wrong gradients, no diagnostic, no crash.

So this fails the build the moment the op starts to look like a real store
while still claiming purity. It is deliberately a *tripwire*, not a design
constraint: the correct response is to drop `Pure`, not to work around this
test. See `docs/audit/compiler/LSE_CHECKPOINT_CONTRACT.md`, and the owning
evaluation rows NVIDIA-LSE-1 / ROCM-LSE-1.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTN_TD = (
    REPO_ROOT
    / "src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td"
)
LOWERING = REPO_ROOT / "src/transforms/lib/TileIRLoweringPass.cpp"

# Operand spellings that mean "this op now writes somewhere".
_DESTINATION_HINTS = (
    "memref", "LLVM_Pointer", "AnyMemRef", "Ptr", "buffer", "workspace",
    "destination", "dest",
)
_EFFECT_HINTS = ("MemWrite", "MemoryEffects", "MemWriteAt", "Effects<")


def _op_body(source: str, op: str) -> str:
    start = source.index(f"def {op} :")
    depth, i = 0, source.index("{", start)
    for end in range(i, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start:end + 1]
    raise AssertionError(f"unterminated ODS body for {op}")


def _traits(body: str) -> str:
    match = re.search(r'def \w+ : Op<[^,]+, "[^"]+", \[([^\]]*)\]', body)
    assert match is not None, f"cannot read traits from:\n{body[:200]}"
    return match.group(1)


def test_lse_save_is_pure_only_while_it_has_no_destination() -> None:
    """A save that can write must not claim purity."""
    body = _op_body(ATTN_TD.read_text(encoding="utf-8"), "LseSaveOp")
    traits = _traits(body)
    if "Pure" not in traits:
        return  # The real contract landed; this tripwire has done its job.

    arguments = body[body.index("let arguments"):body.index("let results")]
    for hint in _DESTINATION_HINTS:
        assert hint not in arguments, (
            f"LseSaveOp gained a destination-shaped operand ({hint!r}) while "
            "still declaring Pure. A store must be non-Pure with MemWrite, or "
            "DCE will silently delete it and the backward will read "
            "uninitialized LSE. Drop Pure — do not relax this test. See "
            "docs/audit/compiler/LSE_CHECKPOINT_CONTRACT.md.")
    for hint in _EFFECT_HINTS:
        assert hint not in body, (
            f"LseSaveOp declares memory effects ({hint!r}) alongside Pure, "
            "which is contradictory. Drop Pure.")


def test_lse_load_still_has_no_source_to_read_from() -> None:
    """Documents that the pair has no identity linking a load to a save.

    When `lse.load` grows a source, the checkpoint has become real and the
    save's trait must be revisited in the same change.
    """
    source = ATTN_TD.read_text(encoding="utf-8")
    body = _op_body(source, "LseLoadOp")
    # The op declares no `let arguments` block at all — not merely an empty
    # one — so there is nothing even shaped like a source to read from.
    if "let arguments" not in body:
        return
    arguments = body[body.index("let arguments"):body.index("let results")]
    if "(ins)" in arguments.replace(" ", ""):
        return
    save_traits = _traits(_op_body(source, "LseSaveOp"))
    assert "Pure" not in save_traits, (
        "lse.load now reads a source, so the checkpoint is real; LseSaveOp "
        "must no longer be Pure. See NVIDIA-LSE-1 / ROCM-LSE-1.")


def test_emission_site_is_still_the_single_destination_less_save() -> None:
    """One emitter, and it still discards the result.

    If the source-level fix lands (stop emitting a destination-less save), this
    test should be deleted along with the Pure marking — not adjusted.
    """
    source = LOWERING.read_text(encoding="utf-8")
    assert source.count('"tessera_attn.lse.save"') == 1, (
        "lse.save now has more than one emission site; the source-level fix in "
        "docs/audit/compiler/LSE_CHECKPOINT_CONTRACT.md assumes exactly one.")
