"""P14 (F5) — vmap (batching) + sharding + transpose contracts for the
diffusion / diffusion_schedule / vision categories.

Before P14 these three categories were absent from the batching / sharding /
transpose rule maps, so their ops resolved to an UNSPECIFIED axis (a `?` in the
coverage scan). P14 formalizes them:

  * diffusion          (edm_loss_weight, edm_precondition) — elementwise in σ →
                       complete on all three axes (the elementwise rule).
  * diffusion_schedule (karras_sigma_schedule, equiprob_band_partition) —
                       deterministic schedule generators from scalar config, no
                       batchable / shardable data axis → explicit
                       axis-specific terminal statuses.
  * vision             (resize / crop / normalize / interpolate) — batch-
                       parallel per-image → the previously-missing sharding rule
                       is complete (batching / transpose were already complete).

This locks the rule values (so a future category-table edit can't silently mis-
claim) and behaviourally verifies the batch-parallel / vmap semantics the rules
assert.
"""

from __future__ import annotations

import numpy as np

from tessera.compiler import primitive_coverage as pc

_COVS = pc.all_primitive_coverages()


def _axes(name: str) -> tuple:
    cs = _COVS[name].contract_status
    return (cs.get("batching_rule"), cs.get("sharding_rule"),
            cs.get("transpose_rule"))


# ── contract-lock ───────────────────────────────────────────────────────────

def test_diffusion_ops_complete_on_all_three_axes():
    for op in ("edm_loss_weight", "edm_precondition"):
        assert _axes(op) == ("complete", "complete", "complete"), op


def test_diffusion_schedule_generators_have_explicit_terminal_contracts():
    for op in ("karras_sigma_schedule", "equiprob_band_partition"):
        assert _axes(op) == (
            "no_batch_axis",
            "replicated_or_non_tensor",
            "no_linear_transpose",
        ), op


def test_vision_ops_sharding_now_complete():
    for op in ("image_resize", "interpolate", "center_crop", "image_normalize"):
        b, s, t = _axes(op)
        assert (b, s, t) == ("complete", "complete", "complete"), op


def test_no_category_is_unspecified():
    """Every primitive's category now resolves in all three rule maps."""
    for m in (pc._BATCHING_RULE_BY_CATEGORY, pc._SHARDING_RULE_BY_CATEGORY,
              pc._TRANSPOSE_RULE_BY_CATEGORY):
        missing = sorted({v.category for v in _COVS.values()
                          if v.category not in m})
        assert not missing, f"categories missing a rule: {missing}"


# ── behavioural: the rules are honest ───────────────────────────────────────

def test_edm_loss_weight_batches_elementwise():
    """batching=complete: w(σ) over a batch of σ equals the per-element map."""
    from tessera.compiler.diffusion_schedule import edm_loss_weight
    sig = np.array([0.1, 0.5, 1.0, 7.0, 80.0])
    batched = np.asarray(edm_loss_weight(sig))
    perelem = np.array([float(edm_loss_weight(float(s))) for s in sig])
    np.testing.assert_allclose(batched, perelem, rtol=1e-12, atol=0)


def test_image_normalize_is_batch_shardable():
    """sharding=complete (vision): splitting the batch axis, normalizing each
    shard, and concatenating reproduces the unsharded result exactly."""
    import tessera
    rng = np.random.default_rng(41)
    x = rng.standard_normal((6, 3, 8, 8)).astype(np.float32)   # NCHW
    mean = np.array([0.1, 0.2, 0.3], np.float32)
    std = np.array([0.5, 0.4, 0.6], np.float32)
    full = np.asarray(tessera.ops.image_normalize(x, mean=mean, std=std))
    shards = [np.asarray(tessera.ops.image_normalize(s, mean=mean, std=std))
              for s in (x[:2], x[2:])]
    np.testing.assert_array_equal(np.concatenate(shards, axis=0), full)
