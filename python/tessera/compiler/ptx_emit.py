"""Phase E1 / rung 2.5 — NVIDIA PTX assembler-text emission (first slice).

See ``docs/audit/compiler/EVALUATOR_PLAN.md`` §2 (rung ladder) and §5.

**The gap this addresses.** Today Tessera's NVIDIA path stops at Target IR MLIR
(``tessera.tile.wgmma`` / ``tessera_nvidia.*``); it emits no assembler text, so
``ptxas`` (rung 3) has nothing to consume and NVIDIA sits at rung 1. This module
is the first step of rung **2.5** — emit real PTX assembler text for a narrow
sm_90a WGMMA bf16 matmul, in the canonical encoding documented in
``docs/nvidia_cuda13_kernel_inventory.md`` and asserted by the
``tests/tessera-ir/phase3/cuda13/`` fixtures.

**Honesty ceiling (read this).** This emits the documented WGMMA *instruction
encoding* inside a structurally-valid PTX kernel skeleton. It is **not** a
complete assemblable kernel: a real WGMMA needs shared-memory matrix descriptors
plus TMA / ``cp.async`` data movement and the full 128-wide accumulator operand
list, which this skeleton deliberately omits (and which cannot be made
ptxas-correct without the toolchain, absent on the arm64 dev host). So:

  * :func:`validate_ptx_structure` checks the PTX scaffolding + that the emitted
    WGMMA mnemonic matches the requested tile/dtype per the CUDA-13.3 inventory —
    verifiable *here*, no toolchain. This is what earns rung 2.5.
  * :func:`ptxas_assemble` invokes real ``ptxas`` (rung 3) — **Linux-CI only**;
    it skip-cleans when the toolchain is absent. It is expected to report what is
    still missing for a complete kernel; assemblability is NOT claimed here.

The complete-kernel work (descriptors + TMA + accumulator vector) is the named
next sub-step; this slice makes "Tessera emits the documented WGMMA PTX
encoding" true and machine-checkable.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# PTX ISA version — imported from the single source of truth (gpu_target.py's
# CUDA pin) so the emitted `.version` directive can never drift from the toolchain
# pin. CUDA Toolkit 13.3 → PTX ISA 9.3.
from .gpu_target import TESSERA_TARGET_PTX_ISA

PTX_ISA_VERSION = TESSERA_TARGET_PTX_ISA

# Documented canonical Hopper WGMMA bf16 tiles (docs/nvidia_cuda13_kernel_inventory.md).
_WGMMA_BF16_CANONICAL: frozenset[tuple[int, int, int]] = frozenset(
    {(64, 256, 16), (64, 128, 16), (64, 64, 16)}
)


def is_valid_wgmma_bf16_shape(m: int, n: int, k: int) -> bool:
    """The Hopper WGMMA bf16 shape constraint: M is fixed at 64, K at 16, and
    N ∈ {8, 16, …, 256} in steps of 8 (PTX ISA 8.x ``wgmma.mma_async`` for
    ``.f32.bf16.bf16``). Stricter than a curated list — it accepts the tile the
    Target IR actually selects (e.g. m64n64k16) and rejects non-WGMMA shapes."""
    return m == 64 and k == 16 and 8 <= n <= 256 and n % 8 == 0


def wgmma_mnemonic(m: int, n: int, k: int, *, acc: str = "f32", ab: str = "bf16") -> str:
    """The documented WGMMA instruction mnemonic for one tile/dtype."""
    return f"wgmma.mma_async.sync.aligned.m{m}n{n}k{k}.{acc}.{ab}.{ab}"


def emit_wgmma_matmul_ptx(
    m: int = 64,
    n: int = 256,
    k: int = 16,
    *,
    arch: str = "sm_90a",
    acc: str = "f32",
    ab: str = "bf16",
    entry: str = "tessera_wgmma_matmul_bf16",
) -> str:
    """Emit a structurally-valid PTX kernel skeleton carrying the canonical
    WGMMA matmul instruction sequence for ``(m, n, k)`` on ``arch``.

    Skeleton only (see module docstring): the WGMMA operands are illustrative,
    not the full descriptor/accumulator operand list. The point is to emit the
    documented *encoding* + the mandatory fence/commit/wait protocol so it can be
    validated and, in CI, fed to ``ptxas``.
    """
    if not is_valid_wgmma_bf16_shape(m, n, k):
        raise ValueError(
            f"({m},{n},{k}) is not a valid Hopper WGMMA bf16 shape "
            "(need m=64, k=16, n∈{8..256 step 8}) — refusing to emit it"
        )
    mma = wgmma_mnemonic(m, n, k, acc=acc, ab=ab)
    return f"""//
// Tessera rung-2.5 emission — {arch} WGMMA {ab} matmul (instruction-encoding skeleton).
// NOT a complete assemblable kernel: real WGMMA needs shared-memory matrix
// descriptors + TMA/cp.async data movement + the full accumulator operand list,
// omitted here. ptxas-assemblability is the rung-3 CI gate; this asserts the
// documented instruction encoding ({mma}) + the PTX scaffolding.
//
.version {PTX_ISA_VERSION}
.target {arch}
.address_size 64

.visible .entry {entry}(
    .param .u64 {entry}_A,
    .param .u64 {entry}_B,
    .param .u64 {entry}_C
)
{{
    .reg .b64  %rd<4>;
    .reg .b64  %desc<2>;
    .reg .f32  %acc<4>;

    ld.param.u64 %rd1, [{entry}_A];
    ld.param.u64 %rd2, [{entry}_B];
    ld.param.u64 %rd3, [{entry}_C];

    // --- WGMMA warpgroup matmul: fence -> mma_async -> commit -> wait ---
    wgmma.fence.sync.aligned;
    {mma} {{%acc0}}, %desc0, %desc1;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    ret;
}}
"""


def validate_ptx_structure(ptx: str, *, arch: str = "sm_90a") -> list[str]:
    """Structural validation of emitted PTX (no toolchain). Returns a list of
    problems — empty means the PTX scaffolding + WGMMA encoding are well-formed.
    This is what earns rung 2.5; it does NOT prove assemblability (rung 3)."""
    problems: list[str] = []
    if f".version {PTX_ISA_VERSION}" not in ptx:
        problems.append(f"missing `.version {PTX_ISA_VERSION}` directive")
    if f".target {arch}" not in ptx:
        problems.append(f"missing `.target {arch}` directive")
    if ".address_size 64" not in ptx:
        problems.append("missing `.address_size 64` directive")
    if ".visible .entry" not in ptx:
        problems.append("no `.visible .entry` kernel")
    if "wgmma.mma_async.sync.aligned.m" not in ptx:
        problems.append("no WGMMA matmul instruction emitted")
    for required in (
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
    ):
        if required not in ptx:
            problems.append(f"missing mandatory WGMMA protocol op `{required}`")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    if "ret;" not in ptx:
        problems.append("kernel does not return (`ret;` missing)")
    return problems


@dataclass(frozen=True)
class AssembleResult:
    """Outcome of a real ``ptxas`` assembly attempt (rung 3)."""

    status: str          # "assembled" | "failed" | "toolchain_absent"
    detail: str = ""

    @property
    def assembled(self) -> bool:
        return self.status == "assembled"


def ptxas_assemble(
    ptx: str, *, arch: str = "sm_90a", ptxas: str | None = None
) -> AssembleResult:
    """Assemble PTX with real ``ptxas`` (rung 3). **Linux-CI only** — returns
    ``toolchain_absent`` (skip-clean) when ptxas is not on PATH, exactly like
    ``scripts/validate_nvcc_compile.py``. Hardware-free: ptxas assembles to SASS
    without a GPU, but it does not install on the arm64 dev host.
    """
    exe = ptxas or shutil.which("ptxas")
    if exe is None:
        return AssembleResult("toolchain_absent", "ptxas not on PATH — skipped")
    try:
        proc = subprocess.run(
            [exe, f"--gpu-name={arch}", "-o", "/dev/null", "-"],
            input=ptx, text=True, capture_output=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return AssembleResult("failed", f"ptxas invocation error: {exc}")
    if proc.returncode == 0:
        return AssembleResult("assembled", "ptxas accepted the kernel")
    return AssembleResult("failed", proc.stderr.strip()[:500])
