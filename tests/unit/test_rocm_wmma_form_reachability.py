"""Every WMMA form this repo declares is either reachable or declared unreachable.

Decision #29: a declaration must have a consumer. `wmma_dtype_forms` enumerates
the matrix instructions each RDNA chip has, and those rows are cited as
coverage -- but enumerating an instruction is not the same as being able to
emit it. Two gaps of exactly that shape were found by hand within a week:

  * RDNA4's double-K int4 (`V_WMMA_I32_16X16X32_IU4`) was declared and
    unreachable because every fragment was pinned to K=16. Closed 2026-09-19.
  * The mixed FP8 pairs (e4m3 x e5m2 and its mirror) were declared and
    unreachable because the single-storage assumption ran the whole depth of
    the stack. Closed 2026-09-19.

Finding those by hand is what this file exists to stop. A form is either
reachable through `lower_scheduled_matmul`, or it is listed below with the
reason, the refusal marker, and the item that owns it.

**The list is checked in three directions, not one.** A form that becomes
reachable while still listed fails. A form that is declared and unreachable
while unlisted fails. And a listed form that is still refused *for a different
reason than recorded* fails too -- that third direction was added 2026-09-19,
after the two reduced-precision-accumulator entries turned out to be
self-fulfilling: the helper answered "unreachable" from the same hardwired rule
that had written the entries, so one named a C++ diagnostic that never fires on
this route and the other was proved by a missing key in a test fixture's own
dtype map. Both had been green since the file was written.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tessera.compiler import scheduled_matmul
from tessera.compiler.rocm_target import AMDArch, wmma_dtype_forms


@dataclass(frozen=True)
class Unreachable:
    """Why the typed route cannot emit a declared form, and how that is checked.

    `reason` is the debt in prose. `marker` is a substring the route's ACTUAL
    refusal must contain, and it is the half that makes this list honest. With
    only prose, "still unreachable" was established by *any* failure at all --
    an import error, a crashed `tessera-opt`, a typo in this file -- so a form
    could stay listed long after its recorded reason stopped applying, and a
    regression that broke it differently would read as the same known gap.
    """

    reason: str
    marker: str


#: (a, b, accum, k) -> the recorded gap. A reason here is a debt, not a
#: dismissal.
UNREACHABLE: dict[tuple[str, str, str, int], Unreachable] = {
    # The mixed OCP FP8 pairs were here until 2026-09-19 and are now REACHABLE:
    # the Schedule carries both operand storages, the fragment and buffer types
    # carry B's own, both C++ gates and the packager admit the pairing, the two
    # launch ABIs are registered, and each operand is validated against its own
    # binding. All four pairings execute natively on gfx1201 with the right
    # instruction. Removing them from this list is the gate doing its job --
    # it fails if a listed form becomes reachable without the list moving.
    ("fp16", "fp16", "fp16", 16): Unreachable(
        reason="f16 accumulation is an opt-in accuracy class behind "
               "tessera.rocm.reduced_precision_accumulation, not a contract the "
               "default route selects. The route refuses in the Python dtype "
               "contract table, BEFORE the backend's own "
               "ROCM_WMMA_ACCUM_UNSUPPORTED can fire -- this entry named that "
               "C++ diagnostic until 2026-09-19 and was wrong, which nothing "
               "noticed because the helper answered `False` without asking.",
        marker="SCHEDULED_MATMUL_DTYPE_CONTRACT_UNSUPPORTED"),
    ("bf16", "bf16", "bf16", 16): Unreachable(
        reason="bf16 accumulation, the same opt-in accuracy class as the f16 "
               "one. Until 2026-09-19 this row was 'proved' unreachable by a "
               "KeyError in the test fixture's own output-element map, so it "
               "was evidence about the fixture and not about the compiler.",
        marker="SCHEDULED_MATMUL_DTYPE_CONTRACT_UNSUPPORTED"),
}

#: The Graph dtype name for a registry storage, where they differ.
_GRAPH_DTYPE = {"fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2"}

#: A form whose K is not 16 is owned by a device row rather than by this gate,
#: because asking for it needs a rewritten descriptor. The owner is named so
#: that renaming or deleting it fails here instead of orphaning the form.
_NON_K16_OWNER = (
    "tests.unit.test_rocm_gfx1201_scheduled",
    "test_gfx1201_double_k_int4_emits_its_instruction_and_is_exact",
)


def _ask_the_route(form, arch_name: str) -> tuple[bool, str]:
    """Ask `lower_scheduled_matmul` for this form. Returns (accepted, refusal).

    Every answer comes from the route. An earlier version of this helper
    short-circuited two cases with a hardcoded `False` -- the same rule that
    put them in UNREACHABLE -- so those entries were self-fulfilling, and the
    same shape had already bitten once: it assumed a mixed fp8 pair was
    inexpressible and had to be taught otherwise before the mixed pairs could
    be retired from the list.
    """
    from tests.unit.test_scheduled_matmul_consumers import _module

    dtype = _GRAPH_DTYPE.get(form.a, form.a)
    b_dtype = _GRAPH_DTYPE.get(form.b, form.b)
    output = {"i32": "int32", "int32": "int32"}.get(form.accum, form.accum)
    try:
        artifact = scheduled_matmul.lower_scheduled_matmul(
            _module(target="rocm", shape=(64, 64, 64), dtype=dtype,
                    b_dtype=b_dtype, output_dtype=output),
            target=f"rocm_{arch_name}")
    except Exception as exc:  # noqa: BLE001 - the text is the evidence
        return False, f"{type(exc).__name__}: {exc}"
    if artifact is None:
        return False, "lower_scheduled_matmul returned None"
    return True, ""


@pytest.mark.parametrize("arch", [AMDArch.GFX_1151, AMDArch.GFX_1201])
def test_every_declared_wmma_form_is_reachable_or_declared_unreachable(arch):
    if scheduled_matmul.find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    arch_name = arch.name.lower().replace("_", "")
    surprises: list[str] = []
    for form in wmma_dtype_forms(arch):
        if form.k != 16:
            continue  # owned by _NON_K16_OWNER, checked below
        key = (form.a, form.b, form.accum, form.k)
        listed = UNREACHABLE.get(key)
        reachable, refusal = _ask_the_route(form, arch_name)
        if reachable and listed:
            surprises.append(
                f"{form.instruction} is listed UNREACHABLE but the route accepts it; "
                f"delete the entry: {listed.reason}")
        elif not reachable and listed is None:
            surprises.append(
                f"{form.instruction} ({form.a} x {form.b} -> {form.accum}) is declared "
                f"by wmma_dtype_forms and nothing can emit it. Either make it "
                f"reachable or add it to UNREACHABLE with the reason, the "
                f"refusal marker, and its item. The route said: {refusal}")
        elif not reachable and listed.marker not in refusal:
            # Still refused, but not for the recorded reason. That is a
            # different gap wearing this one's name.
            surprises.append(
                f"{form.instruction} is still unreachable but NOT for the recorded "
                f"reason. Expected the refusal to contain {listed.marker!r}; got "
                f"{refusal!r}. Update the entry (or fix the regression) -- a stale "
                f"reason is how this list rots into an excuse.")
    assert not surprises, "\n".join(surprises)


def test_a_form_this_gate_delegates_still_has_its_owner():
    """A K != 16 form is skipped above because asking for it needs a rewritten
    descriptor, so a device row owns it instead. If that row is renamed or
    deleted the form is silently owned by nothing -- which is the exact state
    (`declared, emitted by nobody`) this file exists to detect."""
    import importlib

    module_name, test_name = _NON_K16_OWNER
    module = importlib.import_module(module_name)
    assert hasattr(module, test_name), (
        f"{module_name}.{test_name} owns the reachability of every declared "
        f"WMMA form with K != 16 and no longer exists. Either restore it or "
        f"bring those forms into this gate.")
    non_k16 = sorted({f.instruction for arch in (AMDArch.GFX_1151, AMDArch.GFX_1201)
                      for f in wmma_dtype_forms(arch) if f.k != 16})
    assert non_k16, (
        "no declared form has K != 16 any more, so this delegation is dead "
        "code: fold the owner's form back into the gate above and delete this "
        "test with _NON_K16_OWNER")


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
