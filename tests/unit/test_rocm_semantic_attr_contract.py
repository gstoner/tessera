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
    for name in ("ROCM_FloatDTypeAttr", "ROCM_NumericDTypeAttr",
                 "ROCM_GemmDTypeAttr",
                 "ROCM_ReductionAttr", "ROCM_CombineModeAttr",
                 "ROCM_SpectralBackwardKindAttr", "ROCM_ActivationKindAttr",
                 "ROCM_NormKindAttr", "ROCM_Int4PackKindAttr",
                 "ROCM_ReduceKindAttr", "ROCM_ArgReduceKindAttr",
                 "ROCM_ScanKindAttr", "ROCM_UnaryKindAttr",
                 "ROCM_BinaryKindAttr", "ROCM_CompareKindAttr",
                 "ROCM_LogicalKindAttr", "ROCM_BitwiseKindAttr"):
        assert _cases(name), f"{name} declares no cases"


def test_no_semantic_key_is_still_a_bare_strattr():
    """The ratchet: semantic selectors use per-operation closed sets."""
    text = _OPS.read_text()
    offenders = [
        key for key in ("dtype", "reduction", "mode", "kind")
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


def _all_dtype_cases() -> set[str]:
    return (_cases("ROCM_FloatDTypeAttr") | _cases("ROCM_NumericDTypeAttr")
            | _cases("ROCM_GemmDTypeAttr"))


def test_dtype_is_split_by_operation_not_one_shared_union():
    """PR #499 review, P2 — and the reviewer's example was real.

    The first cut declared ONE `ROCM_DTypeAttr` over every dtype any ROCm op
    accepts, so `tessera_rocm.softmax` with `dtype = "int8"` passed ODS and was
    rejected later by `GenerateROCMSoftmaxKernel.cpp`. That is precisely the
    layering W1.1b exists to remove: a real dtype, wrong for that op, admitted
    by the verifier that claims to be fail-closed.

    Measured over the 21 ops carrying `$dtype`: 20 are float-only and exactly
    one (`compare`, which produces a bool from either kind) also takes
    integers. Two constraints, so the wider set is legal for the one op that
    implements it.
    """
    assert _cases("ROCM_FloatDTypeAttr") < _cases("ROCM_NumericDTypeAttr"), (
        "the float set should be a strict subset of the numeric one"
    )
    assert not (_cases("ROCM_FloatDTypeAttr") & {"int8", "i8", "int4", "i4"}), (
        "integer widths are not legal on the float ops"
    )
    assert not (_all_dtype_cases() & {"fp8_e4m3", "fp8_e5m2", "ue4m3", "ue8m0"}), (
        "fp8/microscaling dtypes ride `storage`, never `$dtype`"
    )


def test_the_quant_dtypes_never_belonged_on_dtype():
    """They ride `storage`/`index_storage`, and reached `$dtype` by bad grep.

    `fp8_e4m3`, `int4`, `ue4m3`, `ue8m0` were in the original union because it
    was built by scanning dtype-ish strings across the backend rather than by
    reading what `$dtype` is. No generator branches on any of them from
    `$dtype`. `fpquant` is the case that makes it obvious -- it IS a
    quantisation op, and its `$dtype` is the *input* storage dtype (f32/f16/
    bf16); the target format is `max_normal` / `mantissa_bits` / `min_exp`.
    """
    ops_text = _OPS.read_text()
    for attr in ("ROCM_FloatDTypeAttr", "ROCM_NumericDTypeAttr"):
        assert f"{attr}, " in ops_text or f"{attr}>" in ops_text, (
            f"{attr} is declared but never applied to an op"
        )
    # The narrow-set claim, stated as a value rather than a subset check.
    assert _cases("ROCM_FloatDTypeAttr") == {
        "f32", "float32", "f16", "float16", "bf16", "bfloat16"}


def test_the_gemm_takes_integer_widths_and_the_float_ops_do_not():
    """The three-way split, and the reason a two-way one was wrong.

    `ROCM_WMMAGemmOp` is not an elementwise float op: gfx1151 WMMA has iu8/iu4
    variants, so `int8` and `int4` are real GEMM storage dtypes. The first fix
    for the PR #499 review derived the float set from the `dt == "..."` chains
    in `Generate*Kernel.cpp` -- which the GEMM does not use -- and so narrowed
    the compiled int8/int4 matmul out of existence. It did not fail loudly: the
    artifact stopped verifying and the runtime fell back to the f16/bf16
    executor, and only `test_launch_rocm_compiled_int{4,8}_matches_numpy`
    noticed. Hence this asserts BOTH directions on the same two literals.
    """
    gemm, floats = _cases("ROCM_GemmDTypeAttr"), _cases("ROCM_FloatDTypeAttr")
    for width in ("int8", "i8", "int4", "i4"):
        assert width in gemm, f"the WMMA GEMM lost its {width} lane"
        assert width not in floats, f"{width} must not be legal on softmax et al"
    # The GEMM has no fp32 path -- its ODS default is f16, not f32.
    assert not (gemm & {"f32", "float32"}), (
        "fp32 is absent from the GEMM gate in target_ir.py and from its default"
    )


def test_every_declared_dtype_is_a_real_tessera_dtype_or_a_named_exception():
    """The ODS legal set must not invent dtype spellings.

    Three spellings were in play before this landed, which the bare StrAttr
    hid: the ODS defaults to `f32`/`f16`, the passes accept `f16` OR `float16`
    and `bf16` OR `bfloat16`, and `tessera.dtype`'s canonical names are
    `fp32`/`fp16`/`bf16`. All are admitted so nothing breaks; this gate pins the
    exceptions so the alias set can only shrink.
    """
    from tessera.dtype import canonicalize_dtype

    #: Admitted but NOT canonical. Each needs a reason, and the list may shrink,
    #: never grow -- normalising the tree onto canonical spellings is follow-on.
    exceptions = {
        "f32": "alias for fp32", "float32": "alias for fp32",
        "f16": "alias for fp16", "float16": "alias for fp16",
        "bfloat16": "alias for bf16",
        "i32": "alias for int32", "u32": "alias for uint32",
        "uint32": "uint32 is planned_gated in tessera.dtype",
    }
    unexplained = []
    for value in sorted(_all_dtype_cases()):
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
    how a transitional allowance turns into the contract. Splitting the union
    already shrank it from 13 to 7.
    """
    from tessera.dtype import canonicalize_dtype

    #: Exactly the non-canonical spellings admitted today. Stated in full rather
    #: than derived by a name pattern -- an earlier version tried to recognise
    #: aliases by prefix and mislabelled `int32`, which IS canonical.
    known_aliases = {"f32", "float32", "f16", "float16", "bfloat16",
                     "i32", "u32", "uint32"}
    actual = set()
    for value in _all_dtype_cases():
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
