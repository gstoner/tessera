"""Refuse to report a device lane green when it executed nothing.

Why this exists
---------------
On 2026-08-30 ``pytest tests/device/nvidia/`` on The-Super-Bear reported
**454 passed, 395 skipped, exit 0** while executing zero GPU work. The RTX
5070, ``/dev/dxg`` and CUDA 13.3 were healthy the whole time; only ``PATH``
was wrong, so every device gate probed for ``nvidia-smi``, did not find it,
and skipped. Once the PATH was repaired the same suite surfaced 80 genuine
failures, two of them compiler defects. The suite had been hiding them behind
a green exit code for as long as the misconfiguration lasted.

That was one instance of a shape that showed up five separate times in the
same review campaign: **a check that reports success while having evaluated
nothing.** A device suite that ran no device work, an oracle that reported
agreement over zero comparisons, a ratchet comparing against a baseline
recorded from code that no longer exists. In each case the green was not a
lie about the result -- it was a true statement about an empty set, read as
if it covered a full one.

``pytest.skip`` is the right answer for a host that genuinely lacks the
hardware; that is how a Mac honestly declines to make a CUDA claim, and
CLAUDE.md's Working Rules require exactly that. It is the wrong answer for a
host that *has* the hardware and cannot reach it. This module tells those two
apart and makes only the second one fatal.

What it does NOT do
-------------------
It does not verify that a test which ran actually touched the device -- that
is the job of the per-test ``execution_kind`` assertions (see
``assert_native_gpu``). This is the coarser, complementary claim: that the
lane ran at all. Both are needed. A lane can execute and lie about its
provenance; a lane can also assert provenance perfectly in code that never
runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from tests._support.environment import (
    amx_is_plausibly_present,
    apple_metal4_is_plausibly_present,
    apple_metal_is_plausibly_present,
    avx512_is_plausibly_present,
    nvidia_gpu_is_plausibly_present,
    rocm_gpu_is_plausibly_present,
)


@dataclass(frozen=True)
class DeviceFamily:
    """One hardware lane, its markers, and how to tell it is really here."""

    name: str
    markers: frozenset[str]
    is_plausibly_present: Callable[[], bool]
    remedy: str


#: The hardware markers in ``tests._support.policy.MARKERS`` that denote a
#: physical device, grouped into the lane each one belongs to. A marker absent
#: from every family here is invisible to this gate by construction, which is
#: why ``test_device_accounting.py`` asserts the two stay in step.
DEVICE_FAMILIES: tuple[DeviceFamily, ...] = (
    DeviceFamily(
        name="nvidia",
        markers=frozenset({"hardware_nvidia"}),
        is_plausibly_present=nvidia_gpu_is_plausibly_present,
        remedy=(
            "source scripts/_nvidia_env.sh (the WSL2 driver shim lives in "
            "/usr/lib/wsl/lib and is absent from non-interactive PATH)"
        ),
    ),
    DeviceFamily(
        name="rocm",
        markers=frozenset({"hardware_rocm"}),
        is_plausibly_present=rocm_gpu_is_plausibly_present,
        remedy=(
            "source scripts/_rocm_env.sh (ROCM_PATH and LD_LIBRARY_PATH are "
            "exported only from an interactive .bashrc)"
        ),
    ),
    DeviceFamily(
        name="apple_gpu",
        markers=frozenset({"hardware_apple_gpu"}),
        is_plausibly_present=apple_metal_is_plausibly_present,
        remedy="ninja -C build TesseraAppleRuntimeShared (a stale dylib skips or hangs)",
    ),
    # Metal 4 is tracked SEPARATELY from generic Metal, because folding them
    # into one tally is wrong in both directions. Merged, a single generic
    # Metal test that ran would set executed > 0 and mask a Metal 4 lane that
    # skipped entirely; and a targeted Metal 4 run on an Apple-silicon host
    # whose runtime does not report the capability would skip honestly and be
    # failed by the generic Apple-silicon probe. The probe here is
    # capability-aware for that reason.
    DeviceFamily(
        name="metal4",
        markers=frozenset({"metal4"}),
        is_plausibly_present=apple_metal4_is_plausibly_present,
        remedy=(
            "requires a runtime reporting Metal 4 capability; parts of the "
            "surface are macOS 27.0-gated, so an honest skip here is normal"
        ),
    ),
    DeviceFamily(
        name="avx512",
        markers=frozenset({"hardware_avx512"}),
        is_plausibly_present=avx512_is_plausibly_present,
        remedy=(
            "run on an AVX-512 host (Princess-Luna) with the x86 runtime built: "
            "ninja -C build tessera_x86_elementwise"
        ),
    ),
    # AMX is a dead end, not a pending target: Intel-only, absent from every
    # fleet box, and superseded by ACE (the joint AMD/Intel matrix spec). This
    # family exists solely to give the pre-existing `hardware_amx` marker a
    # consumer; its probe is expected to stay False forever, so the lane can
    # never go hollow. Do not read it as AMX support, and do not gate an
    # AVX-512 test on `hardware_amx` -- that skips it on the only box that
    # could run it.
    DeviceFamily(
        name="amx",
        markers=frozenset({"hardware_amx"}),
        is_plausibly_present=amx_is_plausibly_present,
        remedy=(
            "AMX is not a supported target (superseded by ACE); no fleet host "
            "has it, so this lane is expected to be permanently absent"
        ),
    ),
)


@dataclass
class LaneTally:
    """How many of a family's tests ran, and how many declined to."""

    executed: int = 0
    skipped: int = 0

    @property
    def collected(self) -> int:
        return self.executed + self.skipped


@dataclass
class DeviceLedger:
    """Per-family execution accounting for one pytest session."""

    tallies: dict[str, LaneTally] = field(default_factory=lambda: defaultdict(LaneTally))

    def record(self, families: Iterable[str], *, executed: bool) -> None:
        for name in families:
            tally = self.tallies[name]
            if executed:
                tally.executed += 1
            else:
                tally.skipped += 1

    def hollow_lanes(
        self, families: tuple[DeviceFamily, ...] = DEVICE_FAMILIES
    ) -> list[tuple[DeviceFamily, LaneTally]]:
        """Lanes that skipped everything on a host that appears to have the device.

        All three conditions are load-bearing:

        * ``skipped > 0`` -- a session that collected none of a family's tests
          is not making a claim about it, so there is nothing to falsify. This
          is what keeps ``pytest tests/unit/test_dtype.py`` on the ROCm box
          from failing over untouched GPU lanes.
        * ``executed == 0`` -- one real execution means the lane is reachable
          and the remaining skips are ordinary per-test gating, not a
          host-wide misconfiguration.
        * the device is plausibly present -- otherwise skipping is the honest
          and required answer (Decision #26), not a defect.
        """
        hollow = []
        for family in families:
            tally = self.tallies.get(family.name)
            if tally is None or tally.skipped == 0 or tally.executed:
                continue
            if family.is_plausibly_present():
                hollow.append((family, tally))
        return hollow


def families_for_keywords(
    keywords: Mapping[str, object] | Iterable[str],
    families: tuple[DeviceFamily, ...] = DEVICE_FAMILIES,
) -> frozenset[str]:
    """Map a report's keywords onto the device families it belongs to.

    ``report.keywords`` carries applied markers, so attribution needs no path
    heuristics -- but it follows that an unmarked device test is invisible
    here. That is a real gap rather than a theoretical one: ten files under
    ``tests/device/`` carried no hardware marker as of 2026-08-30.
    """
    present = set(keywords)
    return frozenset(
        family.name for family in families if present & family.markers
    )


def format_hollow_lane_report(
    hollow: list[tuple[DeviceFamily, LaneTally]]
) -> list[str]:
    """Render the failure text. Kept separate so tests can assert on it."""
    lines = [
        "DEVICE LANES SKIPPED ON A HOST THAT HAS THE DEVICE.",
        "",
        "Every test below declined to run, and nothing in the same lane ran at",
        "all -- so this session proves nothing about that hardware. Treating it",
        "as evidence would assert a result that was never produced.",
        "",
    ]
    for family, tally in hollow:
        noun = "test" if tally.skipped == 1 else "tests"
        lines.append(
            f"  {family.name}: 0 executed, {tally.skipped} {noun} skipped"
        )
        lines.append(f"    fix: {family.remedy}")
    lines.extend(
        [
            "",
            "If this host genuinely lacks the hardware, the probe in",
            "tests/_support/environment.py is wrong and should be corrected --",
            "do not silence this by deselecting the lane.",
        ]
    )
    return lines
