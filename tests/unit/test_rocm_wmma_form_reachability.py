"""Every WMMA form this repo declares is either reachable or declared unreachable.

Decision #29: a declaration must have a consumer. `wmma_dtype_forms` enumerates
the matrix instructions each RDNA chip has, and those rows are cited as
coverage -- but enumerating an instruction is not the same as being able to
emit it. Two gaps of exactly that shape were found by hand within a week:

  * RDNA4's double-K int4 (`V_WMMA_I32_16X16X32_IU4`) was declared and
    unreachable because every fragment was pinned to K=16. Closed 2026-09-19.
  * The mixed FP8 pairs (e4m3 x e5m2 and its mirror) are declared and
    unreachable because the Graph matmul contract requires `a_dtype ==
    b_dtype`. Still open.

Finding those by hand is what this file exists to stop. A form is either
reachable through `lower_scheduled_matmul`, or it is listed below with the
reason and the item that owns it -- and a form that becomes reachable without
the list being updated fails here, so the list cannot rot into an excuse.
"""
from __future__ import annotations

import pytest

from tessera.compiler import scheduled_matmul
from tessera.compiler.rocm_target import AMDArch, wmma_dtype_forms

#: (a, b, accum, k) -> why the typed route cannot emit it, and who owns that.
#: A reason here is a debt, not a dismissal.
UNREACHABLE: dict[tuple[str, str, str, int], str] = {
    # Everything BELOW the Schedule is ready as of 2026-09-19: the fragment
    # types carry B's own storage, both the generator gate and the Target
    # matmul gate admit an fp8/fp8 pair, and `resolveFragmentLayout` already
    # selected FP8_BF8 / BF8_FP8 from the descriptor's two types before any of
    # that. What is left is one field: `MatmulSchedule` in PMPasses.cpp carries
    # a single `StringRef storage`, so the Tile IR it emits necessarily writes
    # the same name into the descriptor's `a` and `b`. Until the Schedule can
    # name two, no mixed descriptor can be produced.
    ("fp8_e4m3", "fp8_e5m2", "fp32", 16):
        "mixed FP8 operands: the Schedule carries one storage, so the emitted "
        "descriptor writes the same name to a and b (ROCM-MIXED-FP8-1)",
    ("fp8_e5m2", "fp8_e4m3", "fp32", 16):
        "mixed FP8 operands, mirror of the above (ROCM-MIXED-FP8-1)",
    ("fp16", "fp16", "fp16", 16):
        "f16 accumulation is an opt-in accuracy class behind "
        "tessera.rocm.reduced_precision_accumulation, not a storage the "
        "default route selects (ROCM_WMMA_ACCUM_UNSUPPORTED)",
    ("bf16", "bf16", "bf16", 16):
        "bf16 accumulation, same opt-in accuracy class as the f16 one",
}

#: The Graph dtype name for a registry storage, where they differ.
_GRAPH_DTYPE = {"fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2"}


def _reachable(form, arch_name: str) -> bool:
    """Does `lower_scheduled_matmul` accept a same-dtype matmul in this form?"""
    from tests.unit.test_scheduled_matmul_consumers import _module

    if form.a != form.b:
        return False  # the contract cannot express a mixed pair at all
    dtype = _GRAPH_DTYPE.get(form.a, form.a)
    output = "int32" if form.accum in ("int32", "i32") else "fp32"
    if form.accum in ("fp16", "bf16"):
        return False  # a reduced-precision accumulator, not an output dtype
    shape = (64, 64, 64) if form.k == 16 else (64, 64, 64)
    try:
        artifact = scheduled_matmul.lower_scheduled_matmul(
            _module(target="rocm", shape=shape, dtype=dtype, output_dtype=output),
            target=f"rocm_{arch_name}")
    except Exception:
        return False
    return artifact is not None


@pytest.mark.parametrize("arch", [AMDArch.GFX_1151, AMDArch.GFX_1201])
def test_every_declared_wmma_form_is_reachable_or_declared_unreachable(arch):
    if scheduled_matmul.find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    arch_name = arch.name.lower().replace("_", "")
    surprises: list[str] = []
    for form in wmma_dtype_forms(arch):
        key = (form.a, form.b, form.accum, form.k)
        listed = key in UNREACHABLE
        # The double-K int4 shape is reachable but is not what the production
        # rule selects; `test_gfx1201_double_k_int4_emits_its_instruction_and_is_exact`
        # owns it, and asking for it here would need a rewritten descriptor.
        if form.k != 16:
            continue
        reachable = _reachable(form, arch_name)
        if reachable and listed:
            surprises.append(
                f"{form.instruction} is listed UNREACHABLE but the route accepts it; "
                f"delete the entry: {UNREACHABLE[key]}")
        if not reachable and not listed:
            surprises.append(
                f"{form.instruction} ({form.a} x {form.b} -> {form.accum}) is declared "
                f"by wmma_dtype_forms and nothing can emit it. Either make it "
                f"reachable or add it to UNREACHABLE with the reason and its item.")
    assert not surprises, "\n".join(surprises)


def test_tf32_is_not_a_storage_and_its_rocm_analogue_stays_gated():
    """TF32 is a `math_mode` on fp32, never a storage (Decision #15a), and its
    AMD analogue `mfma_xf32` is CDNA-only.

    The reason to pin this is NVIDIA's NUMPOL-CARRIER-1: that path dispatched
    fp32 storage to a TF32 kernel unconditionally, so a program asking for
    fp32 got TF32 numbers with no diagnostic -- measured up to 1783x the
    relative error. The same shape is available to ROCm the day a CDNA fp32
    lane is built, because `math_mode` has no consumer anywhere in the ROCm
    path. Today there is no hazard on the fleet: RDNA reports xf32
    `not_supported` and the typed route refuses fp32 storage outright, since
    no RDNA WMMA form takes it. This test keeps both halves true.
    """
    from tessera.compiler.rocm_target import rocm_feature_status

    for arch in (AMDArch.GFX_1151, AMDArch.GFX_1201):
        assert rocm_feature_status(arch, "mfma_xf32") == "not_supported"
        assert not any(f.a in ("fp32", "tf32", "xf32") for f in wmma_dtype_forms(arch)), (
            "an RDNA WMMA form claims fp32/tf32 storage; if that is real, "
            "`math_mode` must gate its selection before it ships (see "
            "NUMPOL-CARRIER-1 on the NVIDIA side)")


def test_the_unreachable_list_names_only_forms_that_exist():
    """A stale exception is worse than none: it reads as a known gap for a form
    the hardware may not even have."""
    declared = {(f.a, f.b, f.accum, f.k)
                for arch in (AMDArch.GFX_1151, AMDArch.GFX_1201)
                for f in wmma_dtype_forms(arch)}
    unknown = [k for k in UNREACHABLE if k not in declared]
    assert not unknown, f"UNREACHABLE names forms no chip declares: {unknown}"
