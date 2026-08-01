"""The Apple GPU emitter and the value-lane launcher disagree; bound the gap.

`TileToApple` emits `tessera_apple.gpu.kernel_call {status = "executable"}` when
the named C ABI symbol exists. `runtime.py`'s value-lane launcher
(`_execute_apple_value_target_ir_gpu_artifact`) keeps a shorter allowlist and
raises on anything outside it. Both readings are defensible — the op's `status`
is about the symbol, the launcher's list is about what it has wired up — but the
difference means IR the compiler calls executable can still be refused at launch.

That is not something a verifier should assert (it would reject IR that is
correct by the op's own contract), so it lives here: the delta is allowed to
shrink and never to grow, and a *new* executable GPU op_kind has to either be
launchable or be recorded below with the reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMITTER = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple"
           / "Lowering/TileToApple.cpp")
STREAMING_EMITTER = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple"
                     / "Lowering/StreamingAttentionToAppleGPU.cpp")
RUNTIME = ROOT / "python/tessera/runtime.py"

#: Scope: op_kinds the emitters write as *string literals*. The linalg lane
#: computes its op_kind from a spec table (`sp->graphName` minus the `tessera.`
#: prefix), so kinds like `cholesky` and rank-2 `flash_attn` do not appear here.
#: Those remain follow-up parser work.  The rank-4 GQA consumer writes a literal
#: executable ``flash_attn_gqa`` in the streaming emitter and must stay bound to
#: the value-lane launcher.


def _emitted_executable_gpu_op_kinds() -> set[str]:
    """Literal op_kinds emitted as gpu.kernel_call with status executable."""
    text = EMITTER.read_text(encoding="utf-8")
    kinds: set[str] = set()
    for call in re.findall(r"emitAppleValueCall\((.*?)\);", text, re.S):
        if "tessera_apple.gpu.kernel_call" not in call or '"executable"' not in call:
            continue
        args = [a.strip() for a in re.split(r",(?![^(]*\))", call)]
        if len(args) > 3 and args[3].startswith('"'):
            kinds.add(args[3].strip('"'))
    streaming = STREAMING_EMITTER.read_text(encoding="utf-8")
    if (re.search(r'state\.addAttribute\("op_kind",\s*builder\.getStringAttr\("flash_attn_gqa"\)\)', streaming)
            and 'state.addAttribute("status", builder.getStringAttr("executable"))' in streaming):
        kinds.add("flash_attn_gqa")
    return kinds


def _launcher_allowlist() -> set[str]:
    """The op_kind set literal the GPU value-lane launcher checks against."""
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"), filename=str(RUNTIME))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_execute_apple_value_target_ir_gpu_artifact"):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Compare) and any(
                    isinstance(op, ast.NotIn) for op in inner.ops
                ):
                    operand = inner.comparators[0]
                    if isinstance(operand, ast.Set):
                        return {e.value for e in operand.elts
                                if isinstance(e, ast.Constant)}
    raise AssertionError(
        "could not find the op_kind allowlist in "
        "_execute_apple_value_target_ir_gpu_artifact — did it move?"
    )


def test_launcher_allowlist_is_readable() -> None:
    allowlist = _launcher_allowlist()
    assert "batched_gemm" in allowlist, allowlist
    assert len(allowlist) >= 5, allowlist


def test_no_executable_gpu_op_kind_outruns_the_launcher() -> None:
    emitted = _emitted_executable_gpu_op_kinds()
    assert emitted, "parsed no executable gpu.kernel_call emissions — check the regex"
    gap = sorted(emitted - _launcher_allowlist())
    assert not gap, (
        "TileToApple emits these as executable gpu.kernel_call but the value-lane "
        "launcher rejects them, so launch raises on a compiled artifact: "
        + ", ".join(gap) + ". Wire the launcher, or emit them as status=artifact."
    )


def test_the_launcher_accepts_nothing_the_emitter_cannot_produce() -> None:
    """A launcher entry with no emission is a lane nothing can reach."""
    unreachable = sorted(_launcher_allowlist() - _emitted_executable_gpu_op_kinds())
    assert not unreachable, (
        "the value-lane launcher accepts these op_kinds but TileToApple never "
        "emits them as executable gpu.kernel_call, so no compiled artifact can "
        "reach that branch: " + ", ".join(unreachable)
    )
