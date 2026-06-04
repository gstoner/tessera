from __future__ import annotations

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


def _ppo_artifact(executable: bool = True) -> runtime.RuntimeArtifact:
    return runtime.RuntimeArtifact(
        metadata={
            "target": "apple_gpu",
            "compiler_path": "apple_value_target_ir",
            "executable": executable,
            "arg_names": ["logp_new", "logp_old", "advantages"],
            "apple_value_calls": [{
                "op": "tessera_apple.gpu.kernel_call",
                "op_kind": "ppo_policy_loss",
                "symbol": "tessera_apple_gpu_ppo_policy_loss_f32",
                "status": "executable",
                "clip_epsilon": 0.2,
                "reduction": "mean",
            }],
        },
        abi_signature="tessera.rl.stage13.ppo.apple_gpu",
    )


def test_ppo_value_symbol_is_mpsgraph_backed_not_host_reference():
    source = APPLE_MM.read_text(encoding="utf-8")
    match = re.search(
        r"static bool mpsg_run_ppo_policy_loss_f32[\s\S]*?"
        r"extern \"C\" int32_t tessera_apple_gpu_ppo_policy_loss_f32",
        source,
    )
    assert match, "PPO MPSGraph helper missing"
    body = match.group(0)
    assert "MPSGraph" in body
    assert "runWithMTLCommandQueue" in body
    assert "meanOfTensor" in body
    assert "std::exp" not in body
    assert "for (" not in body


def test_ppo_stub_reports_non_executable_not_host_reference():
    stub = APPLE_STUB.read_text(encoding="utf-8")
    assert "tessera_apple_gpu_ppo_policy_loss_f32" in stub
    assert "return 0;" in stub
    assert "Stage 13 honesty rule" in stub


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
    result = runtime.launch(_ppo_artifact(executable=available), args)
    if not available:
        assert result["ok"] is False
        assert result["runtime_status"] in {
            "unsupported", "invalid_artifact", "unimplemented",
        }
        assert "apple" in result.get("reason", "").lower() or sys.platform != "darwin"
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


def test_ppo_value_launch_rejects_extra_operands_even_if_symbol_claimed():
    artifact = _ppo_artifact(executable=True)
    res = runtime.launch(artifact, [np.zeros((2,), np.float32)] * 4)
    assert res["ok"] is False
    assert res["runtime_status"] == "invalid_artifact"
    assert "exactly 3" in res["reason"]
