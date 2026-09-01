"""The shape axis of applicability: a candidate declining THIS workload.

A region carries structure, not dimensions -- ``MatmulRegion`` has a dtype and
transpose flags, ``FusedRegion`` an epilogue chain, and M/N/K arrive with the
operands. So ``applies_to(region)`` cannot express "aligned shapes only", and
the F4 oracle cannot catch the gap either: its probe shape is fixed (32x16x32
for matmul) and its verdict is cached under a key with no shape in it. A
candidate whose kernel is aligned-only therefore declined *inside* ``run``, by
returning the numpy reference -- after it had already won.

Two consequences, both measured on the real
``NvidiaMmaGemmEmittedCandidate`` before the fix:

* it won arbitration on a ragged shape and handed back numpy while a
  lower-tier lane that could serve the shape went untried;
* ``_measure`` timed that decline and recorded 0.00525 ms of numpy as the
  kernel's latency, against a real 0.00196 ms competitor -- a fabricated
  "2.7x slower" for a kernel that never ran.

Host-free: the ragged decline happens in numpy, above any device call.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera.compiler.fusion as F
from tessera.compiler.emit import autotune as AT
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import OP_MATMUL, Candidate, Tier
from tessera.compiler.emit.nvidia_cuda import NvidiaMmaGemmEmittedCandidate
from tessera.compiler.fusion_core import MatmulRegion

_TGT = "faketarget"

#: M%16 and K%16 both nonzero -> outside the emitted lane's tile alignment.
RAGGED = (24, 12, 20)
#: M%16, N%8, K%16 all zero -> inside it.
ALIGNED = (32, 16, 32)


def _operands(dims: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    m, n, k = dims
    rng = np.random.default_rng(0)
    return (rng.standard_normal((m, k)).astype(np.float32),
            rng.standard_normal((k, n)).astype(np.float32))


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    C._CANDIDATES.clear()
    F.clear_verification_cache()
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)
    F.clear_verification_cache()


class _WorksAnyShape(Candidate):
    """A lower-tier lane with no shape restriction -- the one being starved."""

    name = "generic_any_shape"
    tier = Tier.SYNTHESIZED
    target = _TGT
    op = OP_MATMUL

    def run(self, region, A, B, *a, **k):
        An, Bn = region._natural(A, B)
        return (An @ Bn).astype(np.float32), "generic_real_kernel"


class _DeclinesRaggedInRun(Candidate):
    """Aligned-only, and says so ONLY at run time.

    Deliberately does NOT override ``applies_to_inputs``: this is the candidate
    the fail-closed backstop has to catch, and it stands for every lane that
    has not adopted the hook.
    """

    name = "aligned_only_silent"
    tier = Tier.HAND_TUNED
    target = _TGT
    op = OP_MATMUL

    def run(self, region, A, B, *a, **k):
        An, Bn = region._natural(A, B)
        if An.shape[0] % 16 or Bn.shape[1] % 8 or An.shape[1] % 16:
            return region.reference(A, B), "reference"
        return (An @ Bn).astype(np.float32), "aligned_real_kernel"


class _DeclinesRaggedUpFront(_DeclinesRaggedInRun):
    """The same lane, having adopted the hook."""

    name = "aligned_only_declared"

    def applies_to_inputs(self, region, *inputs):
        if len(inputs) != 2:
            return True
        An, Bn = region._natural(inputs[0], inputs[1], cast=False)
        return not (An.shape[0] % 16 or Bn.shape[1] % 8 or An.shape[1] % 16)


# --- the real candidate ------------------------------------------------------

def test_real_emitted_gemm_declines_a_ragged_workload_it_accepts_by_region():
    """The class documents "aligned only"; until now nothing could ask it."""
    cand = NvidiaMmaGemmEmittedCandidate()
    region = MatmulRegion(dtype="float16")

    # Region-level applicability cannot see the shape -- both answers identical.
    assert cand.applies_to(region) is True

    assert cand.applies_to_inputs(region, *_operands(ALIGNED)) is True
    assert cand.applies_to_inputs(region, *_operands(RAGGED)) is False

    # And the contract it is now stating up front is the one `run` enforces,
    # so the two cannot drift into disagreeing about the same workload.
    _, tag = cand.run(region, *_operands(RAGGED))
    assert tag == "reference"


def test_shape_unknowable_workloads_fail_open_not_closed():
    """Absent operands mean the question cannot be answered, not "no".

    Failing closed here would disable the lane for every caller that arbitrates
    without operands in hand, which is most of them.
    """
    cand = NvidiaMmaGemmEmittedCandidate()
    region = MatmulRegion(dtype="float16")
    assert cand.applies_to_inputs(region) is True
    assert cand.applies_to_inputs(region, np.zeros((4, 4), np.float32)) is True
    assert cand.applies_to_inputs(region, "not", "arrays") is True


def test_rocm_flash_attn_declines_a_head_dim_its_wmma_kernel_cannot_serve():
    """The sibling instance, and the worse-placed one.

    `rocm_flash_attn` is Tier-3 on the one AMD device that executes, so on a
    ragged head_dim it won the dispatch and returned numpy while
    `rocm_generic_hip` stood by. Host-free here: the head_dim arithmetic is
    numpy, above any HIP call. Device evidence for the *dispatch* is owed from
    gfx1151 (Princess-Luna) and is not claimed by this test.
    """
    from tessera.compiler.emit.rocm_hip import RocmFlashAttnCandidate

    cand = RocmFlashAttnCandidate()
    region = F.AttentionRegion()
    rng = np.random.default_rng(0)

    def qkv(head_dim: int):
        return tuple(rng.standard_normal((16, head_dim)).astype(np.float32)
                     for _ in range(3))

    assert cand.applies_to(region) is True            # region-level: blind
    assert cand.applies_to_inputs(region, *qkv(64)) is True
    assert cand.applies_to_inputs(region, *qkv(40)) is False

    # A Q/K head_dim mismatch is an operand error, not an unsupported shape:
    # applicability defers so `run` can report it rather than silently
    # excluding the lane (Decision #21).
    q = rng.standard_normal((16, 64)).astype(np.float32)
    k = rng.standard_normal((16, 32)).astype(np.float32)
    assert cand.applies_to_inputs(region, q, k, k) is True


# --- harm 1: starvation ------------------------------------------------------

def test_a_declining_lane_does_not_starve_a_working_one_on_a_ragged_shape():
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())
    region = MatmulRegion(dtype="float16")
    A, B = _operands(RAGGED)

    winner = C.arbitrate(region, OP_MATMUL, _TGT, verify=False, inputs=(A, B))
    assert winner is not None and winner.name == "generic_any_shape"

    # Not just the name: the dispatch has to actually run a kernel. Selecting
    # the aligned-only lane "succeeds" too -- it returns the right numbers, via
    # numpy, tagged `reference`. That is the silent degrade.
    out, tag = C.run_arbitrated(region, OP_MATMUL, _TGT, A, B,
                                verify=False, use_corpus=False)
    assert tag == "generic_real_kernel"
    np.testing.assert_allclose(out, A @ B, rtol=1e-5, atol=1e-8)


def test_the_aligned_lane_still_wins_the_shape_it_does_serve():
    """The exclusion is workload-scoped, not a demotion."""
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())
    region = MatmulRegion(dtype="float16")
    A, B = _operands(ALIGNED)

    winner = C.arbitrate(region, OP_MATMUL, _TGT, verify=False, inputs=(A, B))
    assert winner is not None and winner.name == "aligned_only_declared"


def test_omitting_inputs_leaves_selection_exactly_as_it_was():
    """The shape axis is additive: no operands, no change in behaviour."""
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())
    region = MatmulRegion(dtype="float16")

    winner = C.arbitrate(region, OP_MATMUL, _TGT, verify=False)
    assert winner is not None and winner.name == "aligned_only_declared"


# --- harm 2: the fabricated measurement --------------------------------------

def _measure_ragged(cache: AT.MeasureCache):
    region = MatmulRegion(dtype="float16")
    A, B = _operands(RAGGED)
    AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B, dims=RAGGED,
                          dtype="float16", cache=cache, reps=3, warmup=1,
                          device="fake:dev")
    return cache.get(("fake:dev", _TGT, OP_MATMUL,
                      AT.bucket_key(RAGGED, AT.SpecPolicy.BUCKET),
                      "float16", AT.TIMING_END_TO_END))


def test_a_run_time_decline_is_recorded_as_unmeasured_not_as_a_latency():
    """The backstop, for a lane that never adopted the hook.

    `_measure` used to time `run` without reading its tag, so a numpy fallback
    was stored under the kernel's name -- and a fabricated latency is worse
    than a missing one, because it ranks.
    """
    C.register_candidate(_DeclinesRaggedInRun())
    C.register_candidate(_WorksAnyShape())

    rec = _measure_ragged(AT.MeasureCache())
    assert rec is not None
    assert rec.winner == "generic_any_shape"
    assert "aligned_only_silent" not in rec.candidates, (
        "a reference decline was timed and stored as this kernel's latency")
    assert rec.unmeasured is not None
    assert "aligned_only_silent" in rec.unmeasured
    assert "declined" in rec.unmeasured["aligned_only_silent"]


def test_a_declared_exclusion_is_absent_from_the_field_not_listed_as_skipped():
    """`unmeasured` means "applicable but not raced" -- keep it that way.

    A candidate that declined the workload was never applicable to it, so
    recording it as a skip would misstate the field in the other direction.
    """
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())

    rec = _measure_ragged(AT.MeasureCache())
    assert rec is not None
    assert rec.winner == "generic_any_shape"
    assert "aligned_only_declared" not in rec.candidates
    assert rec.unmeasured == {}


def test_measured_verdicts_are_not_polluted_across_shapes():
    """The aligned shape still races both lanes and records both latencies."""
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())
    region = MatmulRegion(dtype="float16")
    A, B = _operands(ALIGNED)
    cache = AT.MeasureCache()
    AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B, dims=ALIGNED,
                          dtype="float16", cache=cache, reps=3, warmup=1,
                          device="fake:dev")
    rec = cache.get(("fake:dev", _TGT, OP_MATMUL,
                     AT.bucket_key(ALIGNED, AT.SpecPolicy.BUCKET),
                     "float16", AT.TIMING_END_TO_END))
    assert rec is not None
    assert set(rec.candidates) == {"aligned_only_declared", "generic_any_shape"}


# --- one predicate, every consumer -------------------------------------------

def test_every_race_field_consumer_honours_the_workload_exclusion():
    """`arbitrate`, `measured_arbitrate` and `corpus_winner` each compute "who
    is racing", and each kept its own copy of that filter. Three copies is how
    a rule ends up applied in fewer places than it holds -- so all three must
    agree here, and the check is behavioural rather than a grep for the shared
    helper.
    """
    C.register_candidate(_DeclinesRaggedUpFront())
    C.register_candidate(_WorksAnyShape())
    region = MatmulRegion(dtype="float16")
    A, B = _operands(RAGGED)
    bucket = AT.bucket_key(RAGGED, AT.SpecPolicy.BUCKET)

    # 1. arbitrate
    assert C.arbitrate(region, OP_MATMUL, _TGT, verify=False,
                       inputs=(A, B)).name == "generic_any_shape"

    # 2. measured_arbitrate
    cache = AT.MeasureCache()
    assert AT.measured_arbitrate(region, OP_MATMUL, _TGT, A, B, dims=RAGGED,
                                 dtype="float16", cache=cache, reps=3,
                                 warmup=1, device="fake:dev"
                                 ).name == "generic_any_shape"

    # 3. corpus_winner -- a row whose field was the aligned shape's pair must
    #    still be servable at the ragged shape, where only one lane is live.
    #    Before the shared predicate, `live` here held the excluded lane and
    #    `_record_raced_the_live_field` compared the verdict against a field
    #    that cannot race.
    cache.put(("fake:dev", _TGT, OP_MATMUL, bucket, "float16",
               AT.TIMING_END_TO_END),
              AT.MeasureRecord(winner="generic_any_shape", latency_ms=1.0,
                               candidates={"generic_any_shape": 1.0},
                               unmeasured={}))
    assert AT.corpus_winner(region, OP_MATMUL, _TGT, A, B, dims=RAGGED,
                            dtype="float16", cache=cache,
                            device="fake:dev") == "generic_any_shape"
