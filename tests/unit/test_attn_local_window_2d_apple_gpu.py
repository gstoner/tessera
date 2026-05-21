"""Sub-2 — Apple GPU lowering for ``tessera.attn_local_window_2d``.

Structural + behavioral guards for the new lowering pass that emits a
``func.call @tessera_apple_gpu_attn_local_window_2d_f32`` from the
Graph IR op.  Single-device today; halo-aware distributed paths route
through HaloMeshIntegrationPass first.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLE = REPO_ROOT / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
PASS_CPP = (
    APPLE / "lib" / "Target" / "Apple" / "Lowering"
    / "AttnLocalWindow2DToAppleGPU.cpp"
)
HEADER = APPLE / "include" / "Tessera" / "Target" / "Apple" / "Passes.h"
PIPELINE_CPP = APPLE / "lib" / "Target" / "Apple" / "Passes.cpp"
CMAKE = APPLE / "CMakeLists.txt"
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "attn_local_window_2d_apple_gpu.mlir"
)


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #


def test_pass_source_exists() -> None:
    assert PASS_CPP.exists()


def test_pass_runtime_symbol_name() -> None:
    """The lowering must call the exact runtime symbol the manifest
    plans to back with a fused MSL kernel.  Changes here require a
    matching update in the Apple GPU kernel inventory."""
    text = PASS_CPP.read_text()
    assert "tessera_apple_gpu_attn_local_window_2d_f32" in text


def test_pass_enforces_rank5_f32_envelope() -> None:
    text = PASS_CPP.read_text()
    # The lowering envelope: rank 5, f32, static, patch*D ≤ 1024.
    assert "rank-5" in text
    assert "f32 only" in text
    assert "static shapes" in text
    assert "patch*D > 1024" in text


def test_pass_threads_window_attr() -> None:
    """rh / rw must be read from the Graph IR op's ``window`` attribute
    and threaded into the runtime call as i32 scalars."""
    text = PASS_CPP.read_text()
    assert 'getAttrOfType<ArrayAttr>("window")' in text
    assert "Rhv" in text and "Rwv" in text


def test_pass_uses_falls_through_pattern() -> None:
    """Out-of-envelope inputs must stay in IR (not silently rewritten)
    so the Graph IR op survives to a future fallback."""
    text = PASS_CPP.read_text()
    assert "notifyMatchFailure" in text


def test_create_fn_declared_in_header() -> None:
    text = HEADER.read_text()
    assert "createLowerAttnLocalWindow2DToAppleGPUPass" in text


def test_pass_wired_into_apple_gpu_pipeline() -> None:
    """The Apple GPU runtime pipeline must include this pass between
    the existing window-attention lowerings and the per-op fallbacks."""
    text = PIPELINE_CPP.read_text()
    assert "createLowerAttnLocalWindow2DToAppleGPUPass" in text


def test_cmake_compiles_new_pass() -> None:
    text = CMAKE.read_text()
    assert "AttnLocalWindow2DToAppleGPU.cpp" in text


def test_lit_fixture_covers_static_bf16_asymmetric() -> None:
    text = LIT_FIXTURE.read_text()
    assert "attn_local_window_2d_static_f32" in text
    assert "attn_local_window_2d_bf16_falls_through" in text
    assert "attn_local_window_2d_asymmetric" in text
    assert "tessera-lower-to-apple_gpu-runtime" in text


# --------------------------------------------------------------------------- #
# Behavioral
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


def test_pass_lowers_static_f32_to_runtime_call() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--pass-pipeline=builtin.module(tessera-lower-to-apple_gpu-runtime)",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0 and "Did you mean" in r.stderr:
        pytest.skip("tessera-opt predates the new pass — rebuild required")
    assert r.returncode == 0, f"apple gpu lowering failed: {r.stderr}"
    out = r.stdout

    # The fp32 path emits the runtime call.
    assert "tessera_apple_gpu_attn_local_window_2d_f32" in out
    # And the Graph IR op no longer survives in @attn_local_window_2d_static_f32.
    static_section = out.split("@attn_local_window_2d_bf16_falls_through")[0]
    assert "tessera.attn_local_window_2d" not in static_section, (
        "fp32 path should be fully lowered; got:\n" + static_section[:1000]
    )
    # bf16 path falls through — the op IS still present in the second function.
    bf16_section = out.split("@attn_local_window_2d_bf16_falls_through")[1]
    bf16_section = bf16_section.split("@attn_local_window_2d_asymmetric")[0]
    assert "tessera.attn_local_window_2d" in bf16_section, (
        "bf16 path should stay in IR (out of envelope), got:\n"
        + bf16_section[:600]
    )


def test_runtime_symbol_carries_eleven_arg_signature() -> None:
    """The runtime ABI is locked: 4 i64 pointers + 5 i32 dims + 2 i32
    window half-widths.  If this signature changes, the Apple GPU kernel
    inventory + the manifest must update in lockstep."""
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--pass-pipeline=builtin.module(tessera-lower-to-apple_gpu-runtime)",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        pytest.skip("tessera-opt failure — see test_pass_lowers_static_f32_to_runtime_call")
    # Locate the func.func private decl line.
    for line in r.stdout.splitlines():
        if "tessera_apple_gpu_attn_local_window_2d_f32(" in line and "func.func" in line:
            # Expect: (i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32)
            i64_count = line.count("i64")
            i32_count = line.count("i32")
            assert i64_count == 4, f"expected 4 i64 args, got {i64_count}: {line}"
            assert i32_count == 7, f"expected 7 i32 args, got {i32_count}: {line}"
            return
    pytest.fail("could not find runtime symbol declaration in output")
