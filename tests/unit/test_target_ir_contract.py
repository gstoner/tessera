from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

import tessera as ts
from tessera.compiler.matmul_pipeline import build_cpu_plan, normalize_target_kind
from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.gpu_target import GPUTargetProfile, ISA


def test_target_kind_normalization_accepts_planned_backend_aliases():
    assert normalize_target_kind(None) == "cpu"
    assert normalize_target_kind("cuda") == "nvidia_sm90"
    assert normalize_target_kind("nvidia") == "nvidia_sm90"
    assert normalize_target_kind("gpu") == "nvidia_sm90"
    assert normalize_target_kind("sm90") == "nvidia_sm90"
    assert normalize_target_kind("sm100") == "nvidia_sm100"
    assert normalize_target_kind("sm120") == "nvidia_sm120"
    assert normalize_target_kind(GPUTargetProfile(isa=ISA.SM_100)) == "nvidia_sm100"
    assert normalize_target_kind("hip") == "rocm"
    assert normalize_target_kind("macos_cpu") == "apple_cpu"
    assert normalize_target_kind("m-series-gpu") == "apple_gpu"

    with pytest.raises(ValueError, match="unsupported Tessera target"):
        normalize_target_kind("quantum_waffle")


def test_jit_rocm_target_emits_mfma_and_async_copy_artifact():
    @ts.jit(target="rocm")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    # Target IR text is emitted regardless of the execution lane.
    assert mm.has_target_artifacts
    assert 'target = "rocm"' in mm.target_ir
    assert "tessera_rocm.mfma" in mm.target_ir
    assert "tessera_rocm.async_copy" in mm.target_ir
    assert "tessera_rocm.wait" in mm.target_ir

    artifact = mm.runtime_artifact()
    assert artifact.metadata["target"] == "rocm"
    # Stage L4 — host-gated default flip. On a host with the compiler-generated
    # lane (tessera-opt built + a usable AMD GPU) the rocm matmul executes via
    # `rocm_compiled`; off-device it stays the inspection-only
    # `target_ir_artifact` exactly as before. Assert whichever the host is —
    # never claim executable where it can't run, nor artifact-only where it does.
    from tessera.compiler.jit import _rocm_compiled_lane_available
    if _rocm_compiled_lane_available():
        assert mm.is_executable
        assert artifact.metadata["compiler_path"] == "rocm_compiled"
        assert artifact.metadata["runtime_status"] == "ready"
        assert artifact.metadata["rocm_fallback_lane"] == "rocm_wmma"
    else:
        assert not mm.is_executable
        assert "JIT_TARGET_IR_ARTIFACT_ONLY" in mm.explain_lowering()
        assert artifact.metadata["compiler_path"] == "target_ir_artifact"
        assert artifact.metadata["runtime_status"] == "artifact_only"


def test_jit_hopper_target_emits_wgmma_tma_and_mbarrier_artifact():
    @ts.jit(target=GPUTargetProfile(isa=ISA.SM_90))
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert not mm.is_executable
    assert mm.has_target_artifacts
    assert 'target = "nvidia_sm90"' in mm.target_ir
    assert 'arch = "sm_90a"' in mm.target_ir
    assert "tessera_nvidia.wgmma" in mm.target_ir
    assert "tessera_nvidia.tma_async_copy" in mm.target_ir
    assert "tessera_nvidia.mbarrier" in mm.target_ir

    artifact = mm.runtime_artifact()
    assert artifact.metadata["target"] == "nvidia_sm90"
    assert artifact.metadata["runtime_status"] == "artifact_only"


def test_jit_blackwell_target_emits_tcgen05_and_tmem_artifact():
    @ts.jit(target=GPUTargetProfile(isa=ISA.SM_100))
    def mm(A, B):
        return ts.ops.gemm(A, B)

    assert 'target = "nvidia_sm100"' in mm.target_ir
    assert 'arch = "sm_100a"' in mm.target_ir
    assert "tessera_nvidia.tmem_alloc" in mm.target_ir
    assert "tessera_nvidia.tcgen05_mma" in mm.target_ir
    assert 'block_scaled = true' in mm.target_ir


def test_jit_apple_cpu_target_emits_accelerate_artifact():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert 'target = "apple_cpu"' in mm.target_ir
    assert "tessera_apple.cpu.accelerate_gemm" in mm.target_ir
    assert 'framework = "Accelerate"' in mm.target_ir
    assert 'abi = "cblas_sgemm"' in mm.target_ir


def test_jit_apple_gpu_target_emits_mps_runtime_for_single_matmul():
    """Phase 8.3 — a single rank-2 f32 matmul flips the apple_gpu target IR
    from the artifact-only metal_kernel contract to the MPS runtime contract:
    mps_matmul + mps_dispatch with execution_mode="metal_runtime"."""

    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert 'target = "apple_gpu"' in mm.target_ir
    assert "tessera_apple.gpu.mps_matmul" in mm.target_ir
    assert "tessera_apple.gpu.mps_dispatch" in mm.target_ir
    assert 'execution_mode = "metal_runtime"' in mm.target_ir


def test_jit_apple_gpu_matmul_softmax_chain_emits_fused_msl_runtime_contract():
    """Phase 8.4.3 — the matmul -> softmax chain is the first recognized
    fusion. Both ops collapse into a single fused MSL kernel, flipping the
    program from metal_artifact to metal_runtime."""

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    assert 'target = "apple_gpu"' in fused.target_ir
    assert "tessera_apple.gpu.msl_kernel" in fused.target_ir
    # Optimizing-Compiler Plan F2 — f32 matmul->softmax is SYNTHESIZED (the tiled
    # synthesizer subsumes matmul_softmax_f32 / _tiled_f32).
    assert 'entry_point = "synth_matmul_epi"' in fused.target_ir
    assert 'fusion = "matmul_softmax"' in fused.target_ir
    assert 'execution_mode = "metal_runtime"' in fused.target_ir


def test_jit_apple_gpu_unrecognized_multi_op_keeps_metal_artifact_contract():
    """Multi-op programs that don't match a known fusion pattern stay on
    the artifact path. The recognized fusion set has grown over phases —
    softmax -> matmul (suffix-only, no matmul head) is currently NOT a
    recognized pattern, so it makes a stable negative case."""

    @ts.jit(target="apple_gpu")
    def chain(x, w):
        return ts.ops.matmul(ts.ops.softmax(x), w)

    assert 'target = "apple_gpu"' in chain.target_ir
    assert "tessera_apple.gpu.metal_kernel" in chain.target_ir
    assert "tessera_apple.gpu.dispatch" in chain.target_ir
    assert 'artifact = "metallib"' in chain.target_ir


def test_flash_attention_apple_gpu_gets_msl_runtime_contract():
    """Phase 8.4.1 — single flash_attn programs flip from the artifact-only
    metal_kernel contract to the metal_runtime MSL kernel contract: the
    Target IR carries the MSL source as a StringAttr on
    tessera_apple.gpu.msl_kernel and the runtime executes via a custom
    MTLComputePipelineState."""

    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)

    assert "tessera.flash_attn" in flash.ir_text()
    assert "schedule.pipeline.region" in flash.schedule_ir
    assert "tessera_attn.online_softmax" in flash.tile_ir
    assert "tessera_apple.gpu.msl_kernel" in flash.target_ir
    assert 'entry_point = "flash_attn_f32"' in flash.target_ir
    assert "kernel void flash_attn_f32" in flash.target_ir
    assert 'execution_mode = "metal_runtime"' in flash.target_ir


def test_kv_cache_target_lowering_uses_stable_unsupported_diagnostics():
    source = """
    module decode {
      func step(Cache: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>) -> tensor<?xfp32> {
        C = op.kv_cache_append(Cache, K, V);
        return C;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    plan = build_cpu_plan(module, target_kind="rocm")

    assert plan is not None
    assert "tile.kv_cache" in plan.tile_ir
    assert "tessera.target.diagnostic" in plan.target_ir
    assert 'target = "rocm"' in plan.target_ir
    assert "KV-cache target lowering is not implemented" in plan.target_ir


def test_apple_cpu_matmul_target_executes_accelerate_backend():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(2, dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)

    np.testing.assert_allclose(mm(A, B), A @ B)
    assert mm.is_executable
    artifact = mm.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"


def test_static_target_contract_files_define_backend_spine():
    root = Path(__file__).resolve().parents[2]
    apple = (
        root
        / "src/compiler/codegen/Tessera_Apple_Backend/include/Tessera/Target/Apple/TesseraAppleOps.td"
    ).read_text(encoding="utf-8")
    rocm_passes = (
        root / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp"
    ).read_text(encoding="utf-8")

    assert "cpu.accelerate_gemm" in apple
    assert "gpu.metal_kernel" in apple
    assert "gpu.dispatch" in apple
    assert "tessera-lower-to-rocm" in rocm_passes


# ─────────────────────────────────────────────────────────────────────────────
# W0.9 — real MLIR parse + dialect load + verifier
#
# Every assertion above this line is a SUBSTRING match against Python-generated
# text. That is useful smoke coverage and is deliberately retained, but it can
# never validate Decision #19's actual claim: that each backend exposes a
# hardware-free Target IR *dialect*. A substring test passes just as happily on
# text the dialect's own verifier would reject.
#
# These tests close that gap by running the emitted text through the real
# `tessera-opt`: MLIR parses it, loads the registered dialect, and runs the ODS
# verifiers. A target whose dialect is not compiled into this build is skipped
# rather than failed -- otherwise the result would measure the build config
# instead of the emitter.
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import subprocess
import tempfile


_TESSERA_OPT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt",
)


def _tessera_opt() -> Path | None:
    for candidate in _TESSERA_OPT_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("tessera-opt")
    return Path(found) if found else None


def _registered_dialects(opt: Path) -> set[str]:
    proc = subprocess.run(
        [str(opt), "--show-dialects"], capture_output=True, text=True
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Available Dialects:"):
            line = line.replace("Available Dialects:", "")
        names.update(p.strip() for p in line.split(",") if p.strip())
    return names


def _parse_and_verify(opt: Path, mlir_text: str) -> tuple[bool, str]:
    """Run `mlir_text` through tessera-opt. Returns (ok, first_diagnostic)."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False)
    try:
        handle.write(mlir_text)
        handle.close()
        proc = subprocess.run(
            [str(opt), handle.name, "-o", os.devnull],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(handle.name)
    diagnostic = (proc.stderr.strip().splitlines() or [""])[0]
    return proc.returncode == 0, diagnostic


# target kind → the ODS dialect its Target IR claims to be written in
_TARGET_DIALECT = {
    "cpu": "tessera",
    "x86": "tessera_x86",
    "rocm": "tessera_rocm",
    "apple_cpu": "tessera_apple",
    "apple_gpu": "tessera_apple",
    "nvidia_sm90": "tessera_nvidia",
    "nvidia_sm100": "tessera_nvidia",
    "nvidia_sm120": "tessera_nvidia",
}

# Emitters whose text does NOT survive a real parse + verify today.
#
# Measured 2026-08-02 on a build with tessera_rocm and tessera_apple
# registered. Two distinct defects, both invisible to substring assertions:
#
#   1. Module attributes are not dialect-prefixed (`arch`, `target`,
#      `target_features`), which `builtin.module` rejects outright.
#   2. Underneath that, the ops do not satisfy their own ODS contracts --
#      e.g. `tessera_rocm.mfma` is emitted as `() -> ()` carrying its result
#      as a *string attribute* (`result = "v0"`), while the dialect requires
#      one real SSA result.
#
# So the Python lane emits text that resembles the dialect without being it.
# This is a ratchet: fixing an emitter turns its xfail into an XPASS and forces
# its removal from this list. The list may shrink, never grow.
# EMPTY as of 2026-08-02: every emitter whose dialect is compiled into the
# build now parses, loads its dialect, and passes its ODS verifiers. The three
# original entries (rocm, apple_cpu, apple_gpu) were closed by fixing:
#   * module attributes now dialect-prefixed at MLIR-render time
#     (`_mlir_module_attrs`), keeping the short Python-facing keys;
#   * the function container is `func.func`, not an invented
#     `tessera_apple.cpu.func` / `tessera_rocm.func` that no dialect defines;
#   * `mfma` / `async_copy` / `wait` emit their real ODS signatures, with the
#     async-copy token threaded into the wait.
_KNOWN_UNPARSEABLE: dict[str, str] = {}


def _emit_target_ir(target_kind: str) -> str:
    @ts.jit(target=target_kind)
    def mm(A, B):
        return ts.ops.matmul(A, B)

    return mm.target_ir


@pytest.mark.parametrize("target_kind", sorted(_TARGET_DIALECT))
def test_target_ir_parses_loads_dialect_and_verifies(target_kind):
    """Decision #19's contract, checked by MLIR instead of by `in`.

    A substring test cannot distinguish "emits the dialect" from "emits text
    containing the dialect's name". This one parses the module, loads the
    registered dialect, and runs its ODS verifiers.
    """
    opt = _tessera_opt()
    if opt is None:
        pytest.skip("tessera-opt not built; run `ninja -C build tessera-opt`")

    dialect = _TARGET_DIALECT[target_kind]
    if dialect not in _registered_dialects(opt):
        pytest.skip(
            f"{dialect} is not compiled into this tessera-opt build "
            f"(e.g. -DTESSERA_BUILD_NVIDIA_BACKEND=OFF); skipping so the "
            f"result measures the emitter, not the build config"
        )

    mlir_text = _emit_target_ir(target_kind)
    ok, diagnostic = _parse_and_verify(opt, mlir_text)

    if target_kind in _KNOWN_UNPARSEABLE:
        assert not ok, (
            f"{target_kind} Target IR now parses and verifies. Remove it from "
            f"_KNOWN_UNPARSEABLE — the ratchet may shrink but never grow."
        )
        pytest.xfail(f"{target_kind}: {_KNOWN_UNPARSEABLE[target_kind]} ({diagnostic})")

    assert ok, (
        f"{target_kind} Target IR failed a real MLIR parse/verify:\n"
        f"  {diagnostic}\n"
        f"Decision #19 requires a hardware-free Target IR *dialect*; text that "
        f"only contains the dialect's name does not satisfy it."
    )


def test_known_unparseable_targets_all_have_a_registered_dialect():
    """Guard the ratchet: an entry whose dialect isn't built proves nothing."""
    for target_kind in _KNOWN_UNPARSEABLE:
        assert target_kind in _TARGET_DIALECT, (
            f"{target_kind!r} is not a known target kind"
        )


def test_parse_harness_rejects_invalid_mlir():
    """Guard the guard: a harness that accepts anything measures nothing."""
    opt = _tessera_opt()
    if opt is None:
        pytest.skip("tessera-opt not built")

    ok, _ = _parse_and_verify(opt, "this is definitively not mlir {{{")
    assert not ok, "the parse harness accepted invalid MLIR"

    ok, diagnostic = _parse_and_verify(opt, "module {}")
    assert ok, f"the parse harness rejected a trivially valid module: {diagnostic}"


_GOLDEN_TARGET_IR_DIR = Path(__file__).parent / "golden" / "target_ir"

# golden filename stem suffix → dialect that must be registered to check it
_GOLDEN_SUFFIX_DIALECT = {
    "x86": "tessera_x86",
    "rocm": "tessera_rocm",
    "apple_cpu": "tessera_apple",
    "apple_gpu": "tessera_apple",
    "nvidia_sm90": "tessera_nvidia",
    "nvidia_sm100": "tessera_nvidia",
    "nvidia_sm120": "tessera_nvidia",
}


def _golden_files() -> list[Path]:
    if not _GOLDEN_TARGET_IR_DIR.is_dir():
        return []
    return sorted(_GOLDEN_TARGET_IR_DIR.glob("*.mlir"))


@pytest.mark.parametrize(
    "golden", _golden_files(), ids=lambda p: p.stem
)
def test_committed_golden_target_ir_parses_and_verifies(golden):
    """Every committed Target-IR golden must be real MLIR.

    The single-matmul emitter test above is not sufficient: it exercises one
    lowering path. The `matmul_softmax` goldens are multi-op, and it was
    exactly the *second* op (the softmax → `tessera_rocm.elementwise` path)
    that turned out to emit an operation no dialect declared. Parsing the
    committed goldens covers every path a golden records, for free.
    """
    opt = _tessera_opt()
    if opt is None:
        pytest.skip("tessera-opt not built; run `ninja -C build tessera-opt`")

    suffix = golden.stem.rsplit(".", 1)[-1]

    # NOTE: `cpu` is no longer excluded. The portable reference lane used to
    # mint one op name per source op, which could not be enumerated in ODS; it
    # now emits the single declared `tessera.cpu.reference` node carrying the
    # originating op in its `source` attribute, so it parses and verifies like
    # every other lane.
    dialect = _GOLDEN_SUFFIX_DIALECT.get(suffix)
    if dialect is not None and dialect not in _registered_dialects(opt):
        pytest.skip(f"{dialect} is not compiled into this tessera-opt build")

    ok, diagnostic = _parse_and_verify(opt, golden.read_text())
    assert ok, (
        f"committed golden {golden.name} is not valid MLIR:\n  {diagnostic}\n"
        f"Regenerate with TESSERA_UPDATE_GOLDEN=1 after fixing the emitter."
    )


@pytest.mark.parametrize("target_kind", sorted(_TARGET_DIALECT))
def test_probe_annotated_target_ir_still_parses(target_kind):
    """Probe annotation must preserve typed ops (PR #490 review, P1-3).

    `annotate_target_ir_with_probes()` rebuilds every op through
    `_copy_target_op`. That copy originally dropped the `operand_types` /
    `result_type` / `prelude` fields, which silently downgraded a typed op back
    to `() -> ()` *while keeping its operand references* — so the annotated
    module no longer parsed ("expected 3 operand types but had 0") and its SSA
    operands had no defining prelude.

    Nothing caught it because the plain lowering was the only thing checked.
    This gate covers the annotated path too, for every target.
    """
    from tessera.compiler.frontend import lower_text_to_graph_ir as _lower_graph
    from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
    from tessera.compiler.target_ir import (
        annotate_target_ir_with_probes,
        lower_tile_to_target_ir,
    )
    from tessera.compiler.tile_ir import lower_schedule_to_tile_ir

    opt = _tessera_opt()
    if opt is None:
        pytest.skip("tessera-opt not built; run `ninja -C build tessera-opt`")
    dialect = _TARGET_DIALECT[target_kind]
    if dialect not in _registered_dialects(opt):
        pytest.skip(f"{dialect} is not compiled into this tessera-opt build")

    # Multi-op, so more than one lowering path is exercised.
    source = """
    module demo {
      func main(A: tensor<2x3xfp32>, B: tensor<3x2xfp32>) -> tensor<2x2xfp32> {
        C = op.matmul(A, B);
        P = op.softmax(C);
        return P;
      }
    }
    """
    tile = lower_schedule_to_tile_ir(
        lower_graph_to_schedule_ir(_lower_graph(source))
    )
    module = lower_tile_to_target_ir(tile, target_kind=target_kind)

    ok, diagnostic = _parse_and_verify(opt, module.to_mlir(verify=False))
    assert ok, f"{target_kind} plain Target IR regressed: {diagnostic}"

    annotated = annotate_target_ir_with_probes(module)
    ok, diagnostic = _parse_and_verify(opt, annotated.to_mlir(verify=False))
    assert ok, (
        f"{target_kind} Target IR stopped parsing after probe annotation:\n"
        f"  {diagnostic}\n"
        f"Check that _copy_target_op copies every TargetOp field."
    )
