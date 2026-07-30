"""Calibrated **performance** parameters per target — the missing half of the
target profiles.

``gpu_target.py`` / ``rocm_target.py`` / ``apple_target.py`` describe what a
target *can do* (ISA gates, dtype support, LDS/SMEM capacity, feature matrices).
Nothing in the tree originally described how *fast* it does it: no peak FLOP/s,
DRAM bandwidth, or LLC size. That gap made the old hardware-free cost model
compute-only and made "% of peak" (Theory §8 **W7**) uncomputable. This module
now supplies the explicit inputs consumed by the tile-reuse-distance model.

Two rules make this honest rather than a pile of marketing figures:

**1. Provenance is per field, and never guessed.** Every value carries its own
:class:`Provenance` — ``MEASURED`` (a calibration sweep or on-silicon
confirmation), ``DERIVED`` (an exact identity from a published configuration,
e.g. FP32 peak = lanes x 2 x clock), or ``SPEC`` (vendor sheet). *Per field*
means literally that: one row routinely mixes all three, so a row-level label
would either understate a measured value or overstate a spec one. A field's
provenance comes from :attr:`TargetPerf.field_provenance` when present, else the
row's :attr:`TargetPerf.spec_provenance` default. A value we do not have is
**absent**, not estimated: accessors return ``None`` and :func:`require` raises a
diagnostic naming the gap (Decision #21 — never a silent fabricated default).

**2. Measured beats spec, always.** Vendor sheets overstate. TileSight
(arXiv:2607.22432) Table 3 measures RTX PRO 6000 Blackwell FP32 at 88.6 of 117
"spec" TFLOP/s and DRAM at 1.4 of 1.8 TB/s — a 24% gap on both. A calibration
sweep writes a ``measured`` overlay keyed by field; :meth:`TargetPerf.value`
prefers it and :meth:`TargetPerf.provenance_of` reports which was used, so a
roofline built on spec numbers can never masquerade as one built on silicon.

Leaf module — stdlib only, no Tessera imports — so ``capabilities``,
``backend_manifest``, ``schedule_planner``, the benchmark harness, and the audit
registry can all import it without a cycle.

See ``docs/audit/compiler/TILESIGHT_ASSESSMENT.md`` §2/§4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class Provenance(str, Enum):
    """Where a performance number came from. Ordered worst-to-best by trust."""

    #: Not populated. The accessor returns ``None``; ``require()`` raises.
    UNKNOWN = "unknown"
    #: Vendor spec sheet / marketing figure. Treat as an upper bound only.
    SPEC = "spec"
    #: Exact arithmetic identity from a published configuration — e.g.
    #: ``fp32 peak = cores x 2 flop/clk x boost_clock`` or
    #: ``dram_bw = bus_width_bits / 8 x data_rate``. Reproducible, but still a
    #: theoretical ceiling: no real kernel sustains it.
    DERIVED = "derived"
    #: Measured by a calibration sweep on real silicon. The only kind of number a
    #: performance claim should ultimately rest on.
    MEASURED = "measured"


class TargetPerfError(KeyError):
    """A requested performance field / device is not populated. Names the gap and
    how to fill it rather than returning a fabricated default."""


# --- peak keys ---------------------------------------------------------------
#
# A peak is per (dtype, functional unit): a GPU's fp32 vector rate and its bf16
# matrix-core rate are different pipelines with different ceilings (TileSight
# models them as separate entries of the per-tile resource vector). The key is
# ``"<canonical_dtype>:<unit>"`` — canonical dtype names only (Decision #15a), so
# no ad-hoc ``bf16_tensor`` spellings creep in.

#: Functional units a peak can be quoted against.
UNIT_VECTOR = "vector"   # SIMD / CUDA-core / VALU / CPU SIMD lanes
UNIT_MATRIX = "matrix"   # tensor core / MFMA / WMMA / AMX / simdgroup_matrix
_UNITS = frozenset({UNIT_VECTOR, UNIT_MATRIX})

#: Canonical dtype names accepted in a peak key. Mirrors ``dtype._CANONICAL_DTYPES``
#: restricted to those a FLOP rate is meaningful for; kept literal to preserve the
#: leaf-module property (no ``tessera.dtype`` import).
_PEAK_DTYPES = frozenset({
    "fp64", "fp32", "bf16", "fp16",
    "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4",
    "int8", "int4",
})


def peak_key(dtype: str, unit: str = UNIT_MATRIX) -> str:
    """Build a validated ``"<dtype>:<unit>"`` peak key.

    Raises :class:`ValueError` on a non-canonical dtype or unknown unit — a typo'd
    key would otherwise silently read back as "this target has no such peak".
    """
    if dtype not in _PEAK_DTYPES:
        raise ValueError(
            f"peak dtype {dtype!r} is not a canonical FLOP-rate dtype; "
            f"expected one of {sorted(_PEAK_DTYPES)}")
    if unit not in _UNITS:
        raise ValueError(f"peak unit {unit!r} must be one of {sorted(_UNITS)}")
    return f"{dtype}:{unit}"


#: Scalar (non-peak) fields a ``measured`` overlay may override.
_SCALAR_FIELDS = frozenset({
    "clock_ghz", "dram_bw_gbps", "llc_bytes",
    "compute_units", "smem_bytes_per_cu",
})


def _check_field_name(key: str, *, what: str) -> None:
    """Validate that ``key`` names a real field. A typo'd key would otherwise
    silently have no effect while the spec value kept being reported."""
    if key in _SCALAR_FIELDS:
        return
    if key.startswith("peak_tflops."):
        spec = key.split(".", 1)[1]
        dtype, _, unit = spec.partition(":")
        if not unit:
            raise ValueError(
                f"{what} peak key {key!r} must be "
                f"'peak_tflops.<dtype>:<unit>', e.g. 'peak_tflops.bf16:matrix'")
        peak_key(dtype, unit)  # raises on a bad dtype/unit
        return
    raise ValueError(
        f"unknown {what} field {key!r}; expected one of "
        f"{sorted(_SCALAR_FIELDS)} or 'peak_tflops.<dtype>:<unit>'")


def _validate_measured(measured: Mapping[str, float]) -> None:
    for key in measured:
        _check_field_name(key, what="measured")


@dataclass(frozen=True)
class TargetPerf:
    """Performance parameters for **one physical device**.

    Keyed by device rather than by architecture on purpose: ``nvidia_sm120``
    covers both an RTX 5070 Ti and an RTX PRO 6000 Blackwell, and their peaks
    differ by more than 2x. ``target`` records which canonical
    ``capabilities.normalize_target()`` id the device serves.

    Peaks are in **TFLOP/s** keyed by :func:`peak_key`; bandwidths in **GB/s**
    (10^9 bytes/s, matching vendor convention); capacities in bytes.
    """

    #: Human device name — the registry key. e.g. ``"rtx_5070_ti"``.
    device: str
    #: Canonical ``normalize_target()`` id this device serves.
    target: str
    #: Citation for the spec/derived values. **Required** — an uncited number is
    #: indistinguishable from a guess.
    source: str

    compute_units: int | None = None          # SMs / CUs / GPU cores / CPU cores
    clock_ghz: float | None = None            # boost/sustained clock
    peak_tflops: Mapping[str, float] = field(default_factory=dict)
    dram_bw_gbps: float | None = None
    llc_bytes: int | None = None              # L2 / Infinity Cache / SLC
    smem_bytes_per_cu: int | None = None      # SMEM / LDS / threadgroup memory

    #: Default provenance for populated spec-side values, used for any field with
    #: no :attr:`field_provenance` entry.
    spec_provenance: Provenance = Provenance.SPEC
    #: Per-field provenance overrides. A real row mixes kinds — a derived FLOP
    #: identity next to a spec-sheet core count next to an on-silicon-confirmed
    #: SMEM capacity — and collapsing them to one row label would misreport at
    #: least one of them.
    field_provenance: Mapping[str, Provenance] = field(default_factory=dict)

    #: Calibration overlay: field name -> measured value. Keys are the scalar
    #: field names or ``"peak_tflops.<dtype>:<unit>"``. Always wins over the
    #: spec-side value.
    measured: Mapping[str, float] = field(default_factory=dict)
    #: ISO date of the calibration sweep that produced ``measured``.
    measured_on: str | None = None
    #: Host the sweep ran on, so a corpus row is traceable to a fleet box.
    measured_host: str | None = None
    #: Free-form caveats — dual-issue conventions, cut-down SKUs, shared memory
    #: controllers on an APU, and so on.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.device or not self.target or not self.source:
            raise ValueError("TargetPerf requires device, target, and source")
        for key in self.peak_tflops:
            dtype, _, unit = key.partition(":")
            if not unit:
                raise ValueError(
                    f"peak_tflops key {key!r} must be '<dtype>:<unit>'")
            peak_key(dtype, unit)
        _validate_measured(self.measured)
        for key, prov in self.field_provenance.items():
            _check_field_name(key, what="field_provenance")
            if not isinstance(prov, Provenance):
                raise ValueError(
                    f"field_provenance[{key!r}] must be a Provenance, got {prov!r}")
            if prov is Provenance.UNKNOWN:
                raise ValueError(
                    f"field_provenance[{key!r}] cannot be UNKNOWN — omit the "
                    "field entirely rather than labelling it unknown")
        if self.measured and not self.measured_on:
            raise ValueError(
                f"{self.device}: a measured overlay requires measured_on "
                "(an undated measurement cannot be judged stale)")

    # --- access ---------------------------------------------------------------

    def value(self, field_name: str) -> float | None:
        """Effective value of ``field_name`` — the measured overlay if present,
        else the spec/derived value, else ``None``.

        ``field_name`` is a scalar field name or ``"peak_tflops.<dtype>:<unit>"``.
        """
        if field_name in self.measured:
            return float(self.measured[field_name])
        if field_name.startswith("peak_tflops."):
            got = self.peak_tflops.get(field_name.split(".", 1)[1])
            return None if got is None else float(got)
        if field_name in _SCALAR_FIELDS:
            got = getattr(self, field_name, None)
            return None if got is None else float(got)
        raise TargetPerfError(
            f"unknown TargetPerf field {field_name!r}; expected one of "
            f"{sorted(_SCALAR_FIELDS)} or 'peak_tflops.<dtype>:<unit>'")

    def provenance_of(self, field_name: str) -> Provenance:
        """Which kind of number :meth:`value` would return for ``field_name``.

        Resolution order: the calibration overlay wins; then this field's own
        :attr:`field_provenance` entry; then the row's :attr:`spec_provenance`
        default; ``UNKNOWN`` if the field is not populated at all.
        """
        if field_name in self.measured:
            return Provenance.MEASURED
        if self.value(field_name) is None:
            return Provenance.UNKNOWN
        return self.field_provenance.get(field_name, self.spec_provenance)

    def peak(self, dtype: str, unit: str = UNIT_MATRIX) -> float | None:
        """Peak TFLOP/s for ``(dtype, unit)``, or ``None`` if not populated."""
        return self.value(f"peak_tflops.{peak_key(dtype, unit)}")

    def require(self, field_name: str) -> float:
        """:meth:`value`, but raise a :class:`TargetPerfError` naming the gap and
        the fix instead of returning ``None``. Use wherever a missing number would
        otherwise become a silent zero or a fabricated default."""
        got = self.value(field_name)
        if got is None:
            raise TargetPerfError(
                f"{self.device} ({self.target}) has no {field_name!r}. "
                f"Populate it in target_perf.py with a citation, or run a "
                f"calibration sweep and merge the result via apply_corpus().")
        return got

    # --- derived quantities ---------------------------------------------------

    def ridge_point(self, dtype: str, unit: str = UNIT_MATRIX) -> float | None:
        """Roofline ridge point in **FLOP/byte**: the arithmetic intensity above
        which a kernel is compute-bound. ``None`` if either peak is missing.

        This is the number the current schedule planner cannot compute — it is
        exactly what distinguishes a compute-bound from a memory-bound shape, and
        the reason a FLOPs-only estimator ranks nothing.
        """
        tflops = self.peak(dtype, unit)
        bw = self.value("dram_bw_gbps")
        if tflops is None or bw is None or bw <= 0:
            return None
        return (tflops * 1e12) / (bw * 1e9)

    def roofline_ms(self, flops: float, bytes_moved: float, dtype: str,
                    unit: str = UNIT_MATRIX) -> float | None:
        """Classic roofline lower bound in milliseconds: ``max(compute, memory)``.

        Deliberately minimal — it is a *bound*, not a predictor. It models neither
        pipeline overlap nor cache reuse (``bytes_moved`` is whatever the caller
        says crosses DRAM), so it cannot rank two schedules with the same FLOP and
        byte counts. That is the known limit of roofline and the reason
        TILESIGHT_ASSESSMENT §3.1 T1 exists. Its one job here is to stop the
        estimator from pretending memory traffic is free.
        """
        tflops = self.peak(dtype, unit)
        bw = self.value("dram_bw_gbps")
        if tflops is None or bw is None or tflops <= 0 or bw <= 0:
            return None
        compute_ms = flops / (tflops * 1e12) * 1e3
        memory_ms = bytes_moved / (bw * 1e9) * 1e3
        return max(compute_ms, memory_ms)

    def attainment(self, achieved_tflops: float, dtype: str,
                   unit: str = UNIT_MATRIX) -> float | None:
        """Fraction of peak achieved (**W7**: absolute performance truth).

        ``None`` when the peak is unpopulated — an attainment figure against an
        unknown peak is worse than no figure.
        """
        tflops = self.peak(dtype, unit)
        if tflops is None or tflops <= 0:
            return None
        return achieved_tflops / tflops

    # --- calibration ----------------------------------------------------------

    def with_measured(self, measured: Mapping[str, float], *, on: str,
                      host: str | None = None) -> "TargetPerf":
        """A copy with ``measured`` merged over any existing overlay. Raises on an
        unknown field key so a typo'd sweep fails loudly rather than no-ops."""
        merged = {**self.measured, **{k: float(v) for k, v in measured.items()}}
        _validate_measured(merged)
        return replace(self, measured=merged, measured_on=on,
                       measured_host=host or self.measured_host)

    # --- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device, "target": self.target, "source": self.source,
            "spec_provenance": self.spec_provenance.value,
            "peak_tflops": dict(self.peak_tflops),
        }
        for name in ("compute_units", "clock_ghz", "dram_bw_gbps", "llc_bytes",
                     "smem_bytes_per_cu"):
            got = getattr(self, name)
            if got is not None:
                out[name] = got
        if self.field_provenance:
            out["field_provenance"] = {k: v.value
                                       for k, v in self.field_provenance.items()}
        if self.measured:
            out["measured"] = dict(self.measured)
            out["measured_on"] = self.measured_on
            if self.measured_host:
                out["measured_host"] = self.measured_host
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TargetPerf":
        return cls(
            device=str(d["device"]), target=str(d["target"]),
            source=str(d["source"]),
            compute_units=d.get("compute_units"),
            clock_ghz=d.get("clock_ghz"),
            peak_tflops=dict(d.get("peak_tflops", {})),
            dram_bw_gbps=d.get("dram_bw_gbps"),
            llc_bytes=d.get("llc_bytes"),
            smem_bytes_per_cu=d.get("smem_bytes_per_cu"),
            spec_provenance=Provenance(d.get("spec_provenance", "spec")),
            field_provenance={k: Provenance(v) for k, v
                              in dict(d.get("field_provenance", {})).items()},
            measured=dict(d.get("measured", {})),
            measured_on=d.get("measured_on"),
            measured_host=d.get("measured_host"),
            notes=str(d.get("notes", "")),
        )


# =============================================================================
# The registry
# =============================================================================
#
# Fleet boxes first (the three systems of Theory §6), then reference parts the
# hardware-free lanes are scored against.
#
# DERIVED values use these exact identities:
#   NVIDIA  fp32:vector  = cuda_cores x 2 flop/clk x boost_ghz
#   AMD     fp32:vector  = cus x 64 lanes x 2 flop/clk x boost_ghz
#   Apple   fp32:vector  = gpu_cores x 128 alus x 2 flop/clk x clock_ghz
#   any     dram_bw_gbps = bus_width_bits / 8 x effective_data_rate_gt_s
#
# A dtype with no trustworthy public rate is OMITTED. Calibrate, do not guess.

_REGISTRY: dict[str, TargetPerf] = {}


def register_perf(perf: TargetPerf, *, replace_existing: bool = True) -> None:
    """Register ``perf`` under its ``device``. Idempotent under module re-import."""
    if not replace_existing and perf.device in _REGISTRY:
        raise ValueError(f"device {perf.device!r} already registered")
    _REGISTRY[perf.device] = perf


def _register_all(entries: Iterable[TargetPerf]) -> None:
    for entry in entries:
        register_perf(entry)


_register_all([
    # --- fleet: NR2 Pro (NVIDIA Blackwell consumer) --------------------------
    TargetPerf(
        device="rtx_5070_ti",
        target="nvidia_sm120",
        source=(
            "NVIDIA GeForce RTX 5070 Ti published configuration: 70 SMs / 8960 "
            "CUDA cores, 2.452 GHz boost, 256-bit GDDR7 @ 28 Gbps. SMEM/SM "
            "confirmed on-silicon 2026-06-25 (see gpu_target._SMEM_BYTES)."
        ),
        compute_units=70,
        clock_ghz=2.452,
        peak_tflops={
            # 8960 x 2 x 2.452e9 = 43.9 TFLOP/s
            "fp32:vector": 43.9,
        },
        dram_bw_gbps=896.0,          # 256/8 x 28 = 896
        smem_bytes_per_cu=102400,    # 100 KiB/SM, CC 12.0
        spec_provenance=Provenance.DERIVED,
        field_provenance={
            "compute_units": Provenance.SPEC,
            "clock_ghz": Provenance.SPEC,
            # Confirmed on-silicon 2026-06-25 (RTX 5070 Ti, CUDA 13.3):
            # sharedMemPerMultiprocessor == 102400. Reporting this as merely
            # DERIVED would understate the one value here that touched hardware.
            "smem_bytes_per_cu": Provenance.MEASURED,
        },
        notes=(
            "Matrix peaks intentionally omitted: consumer Blackwell dense "
            "tensor rates depend on accumulate width and the vendor's headline "
            "figure is FP4 *sparse* AI TOPS, which is not a dense matmul peak. "
            "Populate bf16:matrix / fp8_e4m3:matrix from a calibration sweep. "
            "sm_120 has no wgmma and no tcgen05/TMEM (matrix path is mma.sync)."
        ),
    ),

    # --- fleet: Strix Halo (AMD RDNA 3.5 APU) --------------------------------
    TargetPerf(
        device="radeon_8060s",
        target="rocm_gfx1151",
        source=(
            "AMD Ryzen AI Max+ 395 (Strix Halo) integrated Radeon 8060S: 40 "
            "RDNA 3.5 CUs, 2.9 GHz boost, 256-bit LPDDR5X-8000 unified memory, "
            "32 MB MALL (Infinity Cache). LDS/CU from rocm_target._LDS_BYTES."
        ),
        compute_units=40,
        clock_ghz=2.9,
        peak_tflops={
            # 40 x 64 x 2 x 2.9e9 = 14.85 TFLOP/s, single-issue.
            "fp32:vector": 14.85,
        },
        dram_bw_gbps=256.0,          # 256/8 x 8.0 GT/s = 256
        llc_bytes=32 * 1024 * 1024,
        smem_bytes_per_cu=65536,     # 64 KiB LDS/CU (RDNA 3/3.5)
        spec_provenance=Provenance.DERIVED,
        field_provenance={
            "compute_units": Provenance.SPEC,
            "clock_ghz": Provenance.SPEC,
            "llc_bytes": Provenance.SPEC,
            # From the RDNA3.5 ISA guide, not an identity we can rederive.
            "smem_bytes_per_cu": Provenance.SPEC,
        },
        notes=(
            "fp32:vector is the SINGLE-issue rate. RDNA 3.x VOPD dual-issue can "
            "approach 2x on favourable code; quoting the dual-issue number as "
            "'peak' makes every attainment figure look half as good as it is. "
            "Calibrate rather than pick a convention. APU: this DRAM bandwidth "
            "is SHARED with the 16 Zen5 cores (see ryzen_ai_max_395_cpu) — "
            "concurrent CPU traffic reduces what the GPU actually gets, which "
            "is the binding constraint on this box. WMMA is F16/BF16/IU8/IU4; "
            "no FP8/BF8 WMMA on RDNA 3.5."
        ),
    ),
    TargetPerf(
        device="ryzen_ai_max_395_cpu",
        target="x86",
        source=(
            "AMD Ryzen AI Max+ 395 CPU complex: 16 Zen 5 cores / 32 threads, "
            "AVX-512 (no AMX), shared 256-bit LPDDR5X-8000 with the iGPU."
        ),
        compute_units=16,
        dram_bw_gbps=256.0,
        spec_provenance=Provenance.SPEC,
        field_provenance={"dram_bw_gbps": Provenance.DERIVED},  # 256/8 x 8.0 GT/s
        notes=(
            "Peak FLOP/s deliberately UNPOPULATED. Zen 5's AVX-512 throughput "
            "depends on the 512-bit vs double-pumped-256 datapath and on "
            "sustained all-core clocks under AVX-512 load; no published figure "
            "is trustworthy enough to derive from. This entry is the worked "
            "example of the module's rule: an absent number, not a guess. "
            "Fill via calibration. The NR2 Pro's Core Ultra 7 265F is NOT an "
            "x86 lane target — it has neither AVX-512 nor AMX."
        ),
    ),

    # --- fleet: Mac (Apple M1 Max) ------------------------------------------
    TargetPerf(
        device="m1_max",
        target="apple_gpu",
        source=(
            "Apple M1 Max: 32 GPU cores (4096 ALUs) @ 1.296 GHz, 512-bit "
            "LPDDR5-6400 unified memory, 48 MB system level cache."
        ),
        compute_units=32,
        clock_ghz=1.296,
        peak_tflops={
            # 4096 x 2 x 1.296e9 = 10.62 TFLOP/s
            "fp32:vector": 10.62,
        },
        dram_bw_gbps=400.0,          # Apple's published figure
        llc_bytes=48 * 1024 * 1024,
        smem_bytes_per_cu=32768,     # 32 KiB threadgroup memory
        spec_provenance=Provenance.DERIVED,
        field_provenance={
            "compute_units": Provenance.SPEC,
            "clock_ghz": Provenance.SPEC,
            # The 512-bit x 6.4 GT/s identity gives 409.6; Apple publishes 400.
            # This is their number, not ours — SPEC, not DERIVED.
            "dram_bw_gbps": Provenance.SPEC,
            "llc_bytes": Provenance.SPEC,
            "smem_bytes_per_cu": Provenance.SPEC,
        },
        notes=(
            "fp32:vector is DERIVED from the 1.296 GHz clock. Apple publishes "
            "10.4 TFLOPS, which implies ~1.269 GHz — a ~2% clock-convention "
            "gap, not a disagreement about the machine. The derived value is "
            "used so the identity in `source` is checkable; either way this is "
            "a ceiling no kernel reaches, which is what calibration is for. "
            "Apple7 (M1) has native bf16 on the GPU and simdgroup_matrix MMA, "
            "but publishes no matrix-unit FLOP rate — bf16:matrix must be "
            "measured, not derived. Unified memory: this bandwidth is shared "
            "with the CPU. The 48 MB SLC is not an L2 in the discrete-GPU "
            "sense; a cache model must not treat it as one."
        ),
    ),
    TargetPerf(
        device="m1_max_cpu",
        target="apple_cpu",
        source="Apple M1 Max: 8 performance + 2 efficiency cores, shared 400 GB/s.",
        compute_units=10,
        dram_bw_gbps=400.0,
        spec_provenance=Provenance.SPEC,
        notes="Peak FLOP/s unpopulated — calibrate against the Accelerate lane.",
    ),

    # --- reference parts (not in the fleet; hardware-gated lanes) ------------
    TargetPerf(
        device="a100_sxm4_80gb",
        target="nvidia_sm80",
        source=(
            "NVIDIA A100 SXM4 80GB datasheet: 108 SMs, 1.41 GHz boost, 5120-bit "
            "HBM2e. Dense (non-sparse) tensor rates."
        ),
        compute_units=108,
        clock_ghz=1.41,
        peak_tflops={"fp32:vector": 19.5, "bf16:matrix": 312.0,
                     "fp16:matrix": 312.0},
        dram_bw_gbps=2039.0,
        llc_bytes=40 * 1024 * 1024,
        smem_bytes_per_cu=166912,
        spec_provenance=Provenance.SPEC,
        notes=(
            "The 312.0 here is the origin of schedule_planner's hardcoded "
            "peak_tflops=312.0 default, which was being applied to every "
            "target including CPUs. Dense figures; sparse are 2x and must "
            "never be used as a dense matmul ceiling."
        ),
    ),
    TargetPerf(
        device="h100_sxm5",
        target="nvidia_sm90",
        source=(
            "NVIDIA H100 SXM5 datasheet: 132 SMs, 1.98 GHz max clock, HBM3. "
            "Dense (non-sparse) tensor rates."
        ),
        compute_units=132,
        clock_ghz=1.98,
        peak_tflops={"fp32:vector": 67.0, "bf16:matrix": 989.0,
                     "fp16:matrix": 989.0, "fp8_e4m3:matrix": 1979.0},
        dram_bw_gbps=3350.0,
        llc_bytes=50 * 1024 * 1024,
        smem_bytes_per_cu=233472,
        spec_provenance=Provenance.SPEC,
        notes=(
            "TileSight Table 3 measures the H200 sibling at 928 of 989 bf16 "
            "TFLOP/s and 4.2 of 4.8 TB/s — expect a similar spec-to-silicon gap "
            "here. Default clock is 1.83 GHz, not the 1.98 GHz max."
        ),
    ),
    TargetPerf(
        device="mi300x",
        target="rocm_gfx942",
        source=(
            "AMD Instinct MI300X datasheet: 304 CDNA 3 CUs, 2.1 GHz peak "
            "engine clock, 8192-bit HBM3, 256 MB Infinity Cache. Dense rates."
        ),
        compute_units=304,
        clock_ghz=2.1,
        peak_tflops={"fp32:vector": 163.4, "bf16:matrix": 1307.4,
                     "fp16:matrix": 1307.4, "fp8_e4m3:matrix": 2614.9},
        dram_bw_gbps=5300.0,
        llc_bytes=256 * 1024 * 1024,
        smem_bytes_per_cu=65536,
        spec_provenance=Provenance.SPEC,
        notes="Dense figures; sparse are 2x. Hardware-gated lane (Phase G/H).",
    ),
])


# --- lookup ------------------------------------------------------------------

#: Snapshot of the built-in rows, taken before any corpus is applied, so
#: :func:`reset_registry` can undo a calibration merge. Registration mutates
#: process-global state; without a defined way back, a corpus loaded anywhere
#: (a test, a notebook, a CLI) silently colours every later lookup.
_BUILTIN: dict[str, TargetPerf] = dict(_REGISTRY)


def reset_registry() -> None:
    """Restore the built-in rows, discarding every :func:`register_perf` and
    :func:`apply_corpus` mutation made since import.

    The registry is process-global by design (one machine, one set of hardware
    facts), so anything that merges a calibration corpus needs a defined way to
    undo it — otherwise a corpus applied in one place leaks into unrelated code.
    """
    _REGISTRY.clear()
    _REGISTRY.update(_BUILTIN)


def perf_for_device(device: str) -> TargetPerf:
    """The entry for ``device``, or a :class:`TargetPerfError` listing what is
    registered."""
    try:
        return _REGISTRY[device]
    except KeyError:
        raise TargetPerfError(
            f"no performance profile for device {device!r}; registered: "
            f"{sorted(_REGISTRY)}") from None


def devices_for_target(target: str) -> list[TargetPerf]:
    """Every registered device serving canonical target id ``target`` (may be
    empty; may hold several — ``nvidia_sm120`` covers both a 5070 Ti and an RTX
    PRO 6000)."""
    return [p for p in _REGISTRY.values() if p.target == target]


def perf_for_target(target: str, *, device: str | None = None) -> TargetPerf | None:
    """The performance profile for canonical target id ``target``.

    ``None`` when nothing is registered — callers must handle a target with no
    performance data rather than receive an invented one. When several devices
    serve the target, ``device`` disambiguates; without it, a
    :class:`TargetPerfError` names the choices rather than silently picking one
    (picking the wrong sibling is a >2x error).
    """
    matches = devices_for_target(target)
    if device is not None:
        chosen = [p for p in matches if p.device == device]
        if not chosen:
            raise TargetPerfError(
                f"device {device!r} does not serve target {target!r}; "
                f"candidates: {[p.device for p in matches]}")
        return chosen[0]
    if not matches:
        return None
    if len(matches) > 1:
        raise TargetPerfError(
            f"target {target!r} is served by several registered devices "
            f"{[p.device for p in matches]}; pass device= to disambiguate")
    return matches[0]


def registered_devices() -> list[str]:
    """Every registered device name, sorted."""
    return sorted(_REGISTRY)


# --- two JSON artifacts, deliberately distinct -------------------------------
#
# A **calibration corpus** is a sparse overlay of *measured deltas* produced by a
# sweep on one box: `{field: value}` per device, stamped with a date and host.
# A **registry snapshot** is a dense dump of *whole profiles* for committing and
# drift-diffing.
#
# They are not interchangeable, so they do not share a schema. An earlier version
# gave the snapshot the corpus's `version` key and `devices` shape, which made a
# snapshot look loadable by `apply_corpus` when it is not: the rows are full
# profiles, not field overlays, and the required `measured_on` is absent. Each
# now carries a `kind` discriminator so feeding one to the other's loader fails
# by name instead of by a confusing downstream error.

CORPUS_VERSION = 1
SNAPSHOT_VERSION = 1

KIND_CORPUS = "calibration_corpus"
KIND_SNAPSHOT = "registry_snapshot"


def _check_kind(payload: Mapping[str, Any], expected: str, other: str) -> None:
    """Reject a payload that is plainly the *other* artifact. ``kind`` is optional
    for hand-written corpora (the documented schema predates it); when present it
    must match."""
    kind = payload.get("kind")
    if kind is None:
        return
    if kind == other:
        raise ValueError(
            f"this is a {other!r} payload, not a {expected!r}. Registry "
            f"snapshots hold whole profiles and load via "
            f"apply_registry_snapshot(); calibration corpora hold measured "
            f"field overlays and load via apply_corpus().")
    if kind != expected:
        raise ValueError(f"unknown payload kind {kind!r}; expected {expected!r}")


def apply_corpus(corpus: Mapping[str, Any]) -> list[str]:
    """Merge a calibration corpus into the registry. Returns the device names
    updated.

    Schema::

        {"kind": "calibration_corpus",       # optional; checked when present
         "version": 1,
         "measured_on": "2026-07-28",
         "host": "strix-halo",
         "devices": {"radeon_8060s": {"dram_bw_gbps": 214.3,
                                      "peak_tflops.bf16:matrix": 51.2}}}

    A row for an unregistered device, an unknown field, or a mismatched version
    raises — a calibration run that silently lands nowhere is worse than one that
    fails.

    **Atomic.** Every replacement profile is built and validated before any is
    registered, so a corpus whose third row names an unknown device leaves the
    first two untouched. The registry is process-global; a half-applied
    calibration that survives a caught exception would silently colour every
    later lookup with a partial sweep.
    """
    _check_kind(corpus, KIND_CORPUS, KIND_SNAPSHOT)
    version = corpus.get("version")
    if version != CORPUS_VERSION:
        raise ValueError(
            f"calibration corpus version {version!r} != expected "
            f"{CORPUS_VERSION}; regenerate the sweep")
    on = corpus.get("measured_on")
    if not on:
        raise ValueError("calibration corpus requires a 'measured_on' date")
    host = corpus.get("host")
    # Phase 1 — build and validate everything. perf_for_device() raises on an
    # unknown device, with_measured() raises on an unknown field.
    staged: list[TargetPerf] = [
        perf_for_device(device).with_measured(fields, on=str(on), host=host)
        for device, fields in dict(corpus.get("devices", {})).items()
    ]
    # Phase 2 — commit. Nothing below can fail.
    for perf in staged:
        register_perf(perf)
    return [perf.device for perf in staged]


def load_corpus(path: str | Path) -> list[str]:
    """Read a calibration corpus JSON file and :func:`apply_corpus` it."""
    return apply_corpus(_read_json(path, "calibration corpus"))


def registry_snapshot() -> dict[str, Any]:
    """Dump every registered device (spec + any overlay) as whole profiles —
    for committing a fleet-shared snapshot and for drift-diffing.

    Round-trips through :func:`apply_registry_snapshot`. This is **not** a
    calibration corpus: it replaces profiles wholesale rather than overlaying
    measured fields, and `apply_corpus` will refuse it by name.
    """
    return {
        "kind": KIND_SNAPSHOT,
        "version": SNAPSHOT_VERSION,
        "devices": {name: perf.to_dict() for name, perf in sorted(_REGISTRY.items())},
    }


def apply_registry_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    """Replace registered profiles from a :func:`registry_snapshot` payload.
    Returns the device names registered.

    Unlike :func:`apply_corpus` this may introduce devices the running build does
    not know — a snapshot is the whole row, not a delta — which is what makes it
    useful for restoring a fleet-shared table. Atomic for the same reason: every
    row is parsed before any is registered.
    """
    _check_kind(snapshot, KIND_SNAPSHOT, KIND_CORPUS)
    version = snapshot.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"registry snapshot version {version!r} != expected "
            f"{SNAPSHOT_VERSION}; regenerate it")
    rows = dict(snapshot.get("devices", {}))
    staged: list[TargetPerf] = []
    for name, row in rows.items():
        try:
            perf = TargetPerf.from_dict(row)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"registry snapshot row {name!r} is not a valid profile: "
                f"{exc}") from None
        if perf.device != name:
            raise ValueError(
                f"registry snapshot key {name!r} does not match its row's "
                f"device {perf.device!r}")
        staged.append(perf)
    for perf in staged:
        register_perf(perf)
    return [perf.device for perf in staged]


def load_registry_snapshot(path: str | Path) -> list[str]:
    """Read a registry snapshot JSON file and :func:`apply_registry_snapshot` it."""
    return apply_registry_snapshot(_read_json(path, "registry snapshot"))


def _read_json(path: str | Path, what: str) -> Mapping[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text())
    except FileNotFoundError:
        raise TargetPerfError(
            f"{what} {p} not found; generate it first") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} {p} is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{what} {p} must be a JSON object, got {type(payload).__name__}")
    return payload


__all__ = [
    "CORPUS_VERSION",
    "KIND_CORPUS",
    "KIND_SNAPSHOT",
    "Provenance",
    "SNAPSHOT_VERSION",
    "TargetPerf",
    "TargetPerfError",
    "UNIT_MATRIX",
    "UNIT_VECTOR",
    "apply_corpus",
    "apply_registry_snapshot",
    "devices_for_target",
    "load_corpus",
    "load_registry_snapshot",
    "peak_key",
    "registry_snapshot",
    "perf_for_device",
    "perf_for_target",
    "register_perf",
    "registered_devices",
    "reset_registry",
]
