"""Apple GPU loss-family lane — binary-CE / class-axis / RL-policy / EBM.

Parity with the x86/ROCm binary/class/rl/ebm loss lanes: binary_cross_entropy,
cross_entropy / kl / js / z_loss, ppo / cispo / grpo, and the EBM-diffusion
losses reach an executable ``apple_gpu`` path via ``runtime.launch()`` and match
``tessera.losses`` / ``tessera.rl``. The per-sample loss composes through the
standalone reference (host structure); the none/mean/sum reduction runs on the
MPSGraph reduce lane (with a numpy fallback when Metal is unavailable). The
references compute in f64, so the f32 GPU reduction matches at ~2e-4.
"""

from __future__ import annotations

import numpy as np

from tessera import losses, rl
from tessera import runtime as rt

_ATOL = 2e-4
_RTOL = 2e-3


def _launch(op_name, names, args, kwargs=None):
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_gpu_loss_family_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": list(names),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(names),
            "kwargs": dict(kwargs or {}),
        }],
    })
    res = rt.launch(art, args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_loss_family_compiled"
    assert res["execution_kind"] == "native_gpu"
    return res["output"]


def _close(got, ref):
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                               atol=_ATOL, rtol=_RTOL)


def test_apple_gpu_binary_cross_entropy_matches_reference():
    rng = np.random.default_rng(0xB01)
    logits = rng.standard_normal((4, 5)).astype(np.float32)
    targets = rng.integers(0, 2, size=(4, 5)).astype(np.float32)
    for reduction in ("none", "mean", "sum"):
        _close(_launch("tessera.binary_cross_entropy_loss", ("z", "t"),
                       (logits, targets), {"reduction": reduction}),
               losses.binary_cross_entropy_loss(logits, targets,
                                                 reduction=reduction))


def test_apple_gpu_class_axis_losses_match_reference():
    rng = np.random.default_rng(0xB02)
    logits = rng.standard_normal((4, 5)).astype(np.float32)
    targets = rng.integers(0, 5, size=(4,)).astype(np.int64)
    p = np.abs(rng.standard_normal((4, 5))).astype(np.float32)
    p /= p.sum(-1, keepdims=True)
    q = np.abs(rng.standard_normal((4, 5))).astype(np.float32)
    q /= q.sum(-1, keepdims=True)
    for reduction in ("mean", "sum"):
        _close(_launch("tessera.cross_entropy_loss", ("x", "t"),
                       (logits, targets), {"reduction": reduction}),
               losses.cross_entropy_loss(logits, targets, reduction=reduction))
        _close(_launch("tessera.kl_divergence", ("p", "q"),
                       (np.log(p), q), {"reduction": reduction}),
               losses.kl_divergence(np.log(p), q, reduction=reduction))
        _close(_launch("tessera.js_divergence", ("p", "q"), (p, q),
                       {"reduction": reduction}),
               losses.js_divergence(p, q, reduction=reduction))
        _close(_launch("tessera.z_loss", ("x",), (logits,),
                       {"reduction": reduction}),
               losses.z_loss(logits, reduction=reduction))


def test_apple_gpu_rl_policy_losses_match_reference():
    rng = np.random.default_rng(0xB03)
    ln = rng.standard_normal((4, 5)).astype(np.float32)
    lo = rng.standard_normal((4, 5)).astype(np.float32)
    adv = rng.standard_normal((4, 5)).astype(np.float32)
    rew = rng.standard_normal((4, 5)).astype(np.float32)
    for reduction in ("none", "mean", "sum"):
        _close(_launch("tessera.ppo_policy_loss", ("a", "b", "c"),
                       (ln, lo, adv), {"reduction": reduction}),
               rl.ppo_policy_loss(ln, lo, adv, reduction=reduction))
    _close(_launch("tessera.grpo_policy_loss", ("a", "b", "c"), (ln, lo, rew),
                   {"group_axis": 1, "reduction": "mean"}),
           rl.grpo_policy_loss(ln, lo, rew, group_axis=1, reduction="mean"))
    _close(_launch("tessera.cispo_policy_loss", ("a", "b", "c"), (ln, lo, rew),
                   {"group_axis": 1, "epsilon_high": 3.0, "reduction": "sum"}),
           rl.cispo_policy_loss(ln, lo, rew, group_axis=1, epsilon_high=3.0,
                                reduction="sum"))


def test_apple_gpu_ebm_diffusion_losses_match_reference():
    rng = np.random.default_rng(0xB04)
    s = rng.standard_normal((4, 5)).astype(np.float32)
    ts = rng.standard_normal((4, 5)).astype(np.float32)
    yc = rng.standard_normal((4, 5)).astype(np.float32)
    yn = rng.standard_normal((4, 5)).astype(np.float32)
    div = rng.standard_normal((4,)).astype(np.float32)
    ep = rng.standard_normal((4,)).astype(np.float32)
    en = rng.standard_normal((4,)).astype(np.float32)
    terms = rng.standard_normal((4,)).astype(np.float32)
    rp = np.abs(rng.standard_normal((2, 4, 5))).astype(np.float32)
    rp /= rp.sum(-1, keepdims=True)

    _close(_launch("tessera.score_matching_loss", ("a", "b"), (s, ts)),
           losses.score_matching_loss(s, ts))
    _close(_launch("tessera.ddpm_noise_pred_loss", ("a", "b"), (s, ts)),
           losses.ddpm_noise_pred_loss(s, ts))
    _close(_launch("tessera.denoising_score_matching_loss", ("a", "b", "c"),
                   (s, yc, yn), {"sigma": 0.5}),
           losses.denoising_score_matching_loss(s, yc, yn, sigma=0.5))
    _close(_launch("tessera.implicit_score_matching_loss", ("a", "b"), (s, div)),
           losses.implicit_score_matching_loss(s, div))
    for op, fn in (("contrastive_divergence_loss", losses.contrastive_divergence_loss),
                   ("persistent_cd_loss", losses.persistent_cd_loss)):
        _close(_launch(f"tessera.{op}", ("a", "b"), (ep, en)), fn(ep, en))
    for reduction in ("none", "mean", "sum"):
        _close(_launch("tessera.vlb_loss", ("t",), (terms,),
                       {"reduction": reduction}),
               losses.vlb_loss(terms, reduction=reduction))
    _close(_launch("tessera.load_balance_loss", ("rp",), (rp,),
                   {"reduction": "mean"}),
           losses.load_balance_loss(rp, reduction="mean"))
