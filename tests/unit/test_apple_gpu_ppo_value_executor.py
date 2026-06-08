from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from tessera import rl, runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
APPLE_MM = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
)
APPLE_STUB = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp"
)


def _fn_body(source: str, signature_prefix: str) -> str:
    """Return the brace-balanced body of the function whose definition starts
    with ``signature_prefix`` (e.g. ``static bool mpsg_run_ppo_policy_loss_f32``).

    Precise by construction — only the function's own ``{ ... }`` is returned, so
    unrelated code elsewhere in the .mm (e.g. a later kernel that happens to sit
    before the function's ``extern "C"`` wrapper) can't leak into the assertions.
    """
    i = source.find(signature_prefix)
    assert i != -1, f"{signature_prefix!r} not found"
    brace = source.find("{", i)
    assert brace != -1, f"no opening brace after {signature_prefix!r}"
    depth = 0
    for j in range(brace, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[i:j + 1]
    raise AssertionError(f"unbalanced braces for {signature_prefix!r}")


def _metal_runtime_active() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, f"non-darwin host ({sys.platform})"
    try:
        from tessera._apple_gpu_dispatch import apple_gpu_runtime_handle
        handle, path, reason = apple_gpu_runtime_handle()
    except Exception as exc:
        return False, f"apple gpu runtime load failed: {exc}"
    if handle is None:
        return False, reason or "apple gpu runtime unavailable"
    cache_size = getattr(handle, "tessera_apple_gpu_runtime_msl_cache_size", None)
    if cache_size is None:
        return False, f"loaded runtime lacks Metal cache probe: {path}"
    cache_size.argtypes = []
    cache_size.restype = ctypes.c_int32
    if int(cache_size()) < 0:
        return False, f"Metal device/runtime inactive for loaded dylib: {path}"
    return True, f"loaded dylib: {path or 'prebuilt/env runtime'}"


def _ppo_artifact(executable: bool = True, *, extended: bool = False,
                  has_mask: bool = False, has_ref_kl: bool = False,
                  has_entropy: bool = False) -> runtime.RuntimeArtifact:
    arg_names = ["logp_new", "logp_old", "advantages"]
    if has_mask:
        arg_names.append("mask")
    if has_ref_kl:
        arg_names.append("ref_logp")
    if has_entropy:
        arg_names.append("entropy")
    return runtime.RuntimeArtifact(
        metadata={
            "target": "apple_gpu",
            "compiler_path": "apple_value_target_ir",
            "executable": executable,
            "arg_names": arg_names,
            "apple_value_calls": [{
                "op": "tessera_apple.gpu.kernel_call",
                "op_kind": "ppo_policy_loss",
                "symbol": (
                    "tessera_apple_gpu_ppo_policy_loss_ex_f32" if extended
                    else "tessera_apple_gpu_ppo_policy_loss_f32"),
                "status": "executable",
                "clip_epsilon": 0.2,
                "kl_coef": 0.03 if has_ref_kl else 0.0,
                "entropy_coef": 0.02 if has_entropy else 0.0,
                "has_mask": has_mask,
                "has_ref_kl": has_ref_kl,
                "has_entropy": has_entropy,
                "reduction": "mean",
            }],
        },
        abi_signature="tessera.rl.stage14.ppo.apple_gpu",
    )


def test_ppo_value_symbol_is_mpsgraph_backed_not_host_reference():
    source = APPLE_MM.read_text(encoding="utf-8")
    # The extern "C" wrapper must exist (the symbol is shipped)...
    assert "extern \"C\" int32_t tessera_apple_gpu_ppo_policy_loss_f32" in source
    # ...and the MPSGraph helper it calls must be MPSGraph-backed (no host loop).
    body = _fn_body(source, "static bool mpsg_run_ppo_policy_loss_f32")
    assert "MPSGraph" in body
    assert "runWithMTLCommandQueue" in body
    assert "meanOfTensor" in body
    assert "std::exp" not in body
    assert "for (" not in body


def test_extended_ppo_value_symbol_is_mpsgraph_backed_not_host_reference():
    source = APPLE_MM.read_text(encoding="utf-8")
    assert "extern \"C\" int32_t tessera_apple_gpu_ppo_policy_loss_ex_f32" in source
    body = _fn_body(source, "static bool mpsg_run_ppo_policy_loss_ex_f32")
    assert "MPSGraph" in body
    assert "runWithMTLCommandQueue" in body
    assert "reductionSumWithTensor" in body
    assert "meanOfTensor" in body
    assert "std::exp" not in body
    assert "for (" not in body


def test_ppo_stub_reports_non_executable_not_host_reference():
    stub = APPLE_STUB.read_text(encoding="utf-8")
    assert "tessera_apple_gpu_ppo_policy_loss_f32" in stub
    assert "tessera_apple_gpu_ppo_policy_loss_ex_f32" in stub
    assert "return 0;" in stub
    assert "Stage 13 honesty rule" in stub
    assert "Stage 14 honesty rule" in stub


def test_ppo_value_launch_matches_reference_when_available():
    rng = np.random.default_rng(13)
    logp_old = rng.normal(-1.0, 0.2, size=(2, 3, 5)).astype(np.float32)
    logp_new = (logp_old + rng.normal(0.0, 0.04, size=(2, 3, 5))).astype(np.float32)
    advantages = rng.normal(0.0, 1.0, size=(2, 3, 5)).astype(np.float32)
    args = {
        "logp_new": logp_new,
        "logp_old": logp_old,
        "advantages": advantages,
    }
    available = runtime._apple_gpu_ppo_policy_loss_available()
    active, detail = _metal_runtime_active()
    if active and not available:
        pytest.fail(
            "Active Apple GPU runtime did not pass the PPO MPSGraph value "
            "probe. Rebuild/refresh the shared dylib and run in a fresh Python "
            "process; expected _apple_gpu_ppo_policy_loss_available() == True. "
            f"{detail}")
    result = runtime.launch(_ppo_artifact(executable=available), args)
    if not available:
        assert result["ok"] is False
        assert result["runtime_status"] in {
            "unsupported", "invalid_artifact", "unimplemented",
        }
        assert "apple" in result.get("reason", "").lower() or not active
        return
    expected = rl.ppo_policy_loss(
        logp_new,
        logp_old,
        advantages,
        clip_epsilon=0.2,
        reduction="mean",
    )
    assert result["ok"] is True
    assert result["compiler_path"] == "apple_value_target_ir"
    assert np.allclose(np.asarray(result["output"]), expected, rtol=1e-4, atol=1e-5)


def test_extended_ppo_value_launch_matches_reference_when_available():
    rng = np.random.default_rng(14)
    logp_old = rng.normal(-1.0, 0.2, size=(2, 3, 5)).astype(np.float32)
    logp_new = (logp_old + rng.normal(0.0, 0.04, size=(2, 3, 5))).astype(np.float32)
    advantages = rng.normal(0.0, 1.0, size=(2, 3, 5)).astype(np.float32)
    mask = (rng.random(size=(2, 3, 5)) > 0.25).astype(np.float32)
    ref_logp = (logp_old + rng.normal(0.0, 0.03, size=(2, 3, 5))).astype(np.float32)
    entropy = np.abs(logp_new).astype(np.float32) * 0.1
    args = {
        "logp_new": logp_new,
        "logp_old": logp_old,
        "advantages": advantages,
        "mask": mask,
        "ref_logp": ref_logp,
        "entropy": entropy,
    }
    available = runtime._apple_gpu_ppo_policy_loss_ex_available()
    active, detail = _metal_runtime_active()
    if active and not available:
        pytest.fail(
            "Active Apple GPU runtime did not pass the extended PPO MPSGraph "
            "value probe. Rebuild/refresh the shared dylib and run in a fresh "
            "Python process; expected "
            "_apple_gpu_ppo_policy_loss_ex_available() == True. "
            f"{detail}")
    artifact = _ppo_artifact(
        executable=available, extended=True, has_mask=True, has_ref_kl=True,
        has_entropy=True)
    result = runtime.launch(artifact, args)
    if not available:
        assert result["ok"] is False
        assert result["runtime_status"] in {
            "unsupported", "invalid_artifact", "unimplemented",
        }
        return
    expected = rl.ppo_policy_loss(
        logp_new,
        logp_old,
        advantages,
        clip_epsilon=0.2,
        mask=mask,
        ref_logp=ref_logp,
        kl_coef=0.03,
        entropy=entropy,
        entropy_coef=0.02,
        reduction="mean",
    )
    assert result["ok"] is True
    assert np.allclose(np.asarray(result["output"]), expected, rtol=1e-4, atol=1e-5)


def test_ppo_value_launch_rejects_extra_operands_even_if_symbol_claimed():
    artifact = _ppo_artifact(executable=True)
    res = runtime.launch(artifact, [np.zeros((2,), np.float32)] * 4)
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "exactly 3" in res["reason"]
