"""W1.1b — semantic string attributes in the ROCm dialect state their legal set.

Decision #21a: an attribute that selects SEMANTICS must fail closed. Every ROCm
kernel op took a bare ``StrAttr`` for ``dtype`` / ``reduction`` / ``mode``, and
the passes read them as free strings — an unrecognised value fell through a
chain of ``==`` comparisons to whatever the last ``else`` happened to do.

**The plan item's count was wrong, and acting on it would have been a defect.**
W1.1b is written as "62 x ``$name``", but ``$name`` is the emitted KERNEL
SYMBOL: measured values include ``"flash"``, ``"fc1"``, ``"fc2"``, ``"bwd"``,
``"acc"``, ``"default_int4_wmma"``. That is an open set chosen by the caller,
not a semantic selector, and enumerating it would be incorrect. The genuinely
semantic attributes are ``dtype`` (24 ops), ``kind`` (14), ``reduction`` (3),
``mode``, ``storage``, ``route``, ``activation``, ``feature_map``, ``counter``.

This file gates the ODS declarations against the rest of the compiler, since a
legal set written in one place and consumed in another is exactly what the PB
thread kept finding broken.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DIALECT = (_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/include"
            / "TesseraROCM/IR/TesseraROCMDialect.td")
_OPS = (_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/include"
        / "TesseraROCM/IR/TesseraROCMOps.td")


def _cases(attr_name: str) -> set[str]:
    """The legal set declared for `attr_name` in the dialect .td."""
    text = _DIALECT.read_text()
    match = re.search(rf"def {attr_name}\s*:\s*ROCM_EnumStrAttr<\[(.*?)\]",
                      text, re.S)
    assert match, f"{attr_name} is not declared as a ROCM_EnumStrAttr"
    # Strip `//` comments first: the case lists are annotated inline, and those
    # comments contain quoted phrases ("no dtype") that a bare quote-scan reads
    # as cases. The first run of this gate reported exactly that.
    body = "\n".join(line.split("//")[0] for line in match.group(1).splitlines())
    return set(re.findall(r'"([^"]+)"', body))


def test_the_semantic_attrs_are_declared_with_a_legal_set():
    for name in ("ROCM_DTypeAttr", "ROCM_ReductionAttr", "ROCM_CombineModeAttr"):
        assert _cases(name), f"{name} declares no cases"


def test_no_semantic_key_is_still_a_bare_strattr():
    """The ratchet: `dtype`/`reduction`/`mode` must not regress to `StrAttr`.

    Scoped to these three deliberately. `kind` (14 ops) is NOT included because
    each op uses it for a DIFFERENT closed set — optimizer kinds, predicate
    kinds, GA product kinds — so one shared enum would be wrong and per-op
    enums are separate work. Claiming it here would make the gate assert
    something untrue.
    """
    text = _OPS.read_text()
    offenders = [
        key for key in ("dtype", "reduction", "mode")
        if f"StrAttr:${key}" in text
        or re.search(rf"DefaultValuedAttr<StrAttr,[^>]*>:\${key}", text)
    ]
    assert not offenders, (
        f"semantic keys still declared as bare StrAttr: {offenders}"
    )


def test_name_is_deliberately_left_as_a_free_string():
    """`$name` is the kernel SYMBOL, not a semantic selector.

    Asserted positively so the next person reading the plan item ("62 x
    $name") does not convert it. An enum over caller-chosen symbol names would
    reject every valid program that picks a name the enum did not anticipate.
    """
    text = _OPS.read_text()
    assert "StrAttr:$name" in text, (
        "$name should remain a free string — it is the emitted kernel symbol "
        "(measured values: flash, fc1, fc2, bwd, acc, default_int4_wmma)"
    )


def test_every_declared_dtype_is_a_real_tessera_dtype_or_a_named_exception():
    """The ODS legal set must not invent dtype spellings.

    Three spellings were in play before this landed, which the bare StrAttr
    hid: the ODS defaulted to `f32`/`f16`, the passes accepted `f16` OR
    `float16` and `bf16` OR `bfloat16`, and `tessera.dtype`'s canonical names
    are `fp32`/`fp16`/`bf16`. All are admitted for now so nothing breaks; this
    gate pins the exceptions so the alias set can only shrink.
    """
    from tessera.dtype import canonicalize_dtype

    #: Admitted but NOT canonical. Each needs a reason, and the list may shrink,
    #: never grow — normalising the tree onto canonical spellings is follow-on.
    exceptions = {
        # Seven alias pairs the ROCm passes accept today.
        "f32": "alias for fp32", "float32": "alias for fp32",
        "f16": "alias for fp16", "float16": "alias for fp16",
        "bfloat16": "alias for bf16",
        "i8": "alias for int8", "i4": "alias for int4",
        "i32": "alias for int32", "u32": "alias for uint32",
        "uint32": "uint32 is planned_gated in tessera.dtype",
        "ue4m3": "microscaling SCALE dtype, not a storage dtype",
        "ue8m0": "microscaling SCALE dtype, not a storage dtype",
        "none": "explicit 'no dtype' for ops that key off operand types",
    }
    unexplained = []
    for value in sorted(_cases("ROCM_DTypeAttr")):
        if value in exceptions:
            continue
        try:
            canonicalize_dtype(value)
        except Exception:
            unexplained.append(value)
    assert not unexplained, (
        f"ODS declares dtype spellings that are neither canonical nor a named "
        f"exception: {unexplained}"
    )


def test_the_alias_set_is_a_shrink_only_ratchet():
    """Aliases may be removed, never added.

    Without this the "admitted for now" exceptions become permanent, which is
    how a transitional allowance turns into the contract.
    """
    from tessera.dtype import canonicalize_dtype

    #: Exactly the non-canonical spellings admitted today. Stated in full
    #: rather than derived by a name pattern -- the first version tried to
    #: recognise aliases by prefix and mislabelled `int8`/`int32`, which ARE
    #: canonical. A ratchet that cannot say precisely what it guards is not one.
    known_aliases = {
        "f32", "float32", "f16", "float16", "bfloat16",
        "i8", "i4", "i32", "u32", "uint32",
        "ue4m3", "ue8m0", "none",
    }
    actual = set()
    for value in _cases("ROCM_DTypeAttr"):
        try:
            canonicalize_dtype(value)
        except Exception:
            actual.add(value)
    new = actual - known_aliases
    assert not new, f"new non-canonical dtype spellings introduced: {sorted(new)}"


@pytest.mark.parametrize(
    ("attr", "expected"),
    [("ROCM_ReductionAttr", {"none", "sum", "mean"}),
     ("ROCM_CombineModeAttr", {"add", "max", "min", "set", "weighted_add"})],
)
def test_small_closed_sets_are_exactly_what_the_passes_accept(attr, expected):
    """These two sets are small enough to state exhaustively, so state them.

    `reduction` in particular is worth pinning: `median`, `logsumexp` and
    `softmin` are all real reductions elsewhere in the tree, so a plausible
    value this kernel cannot perform was previously accepted in silence.
    """
    assert _cases(attr) == expected
