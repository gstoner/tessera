"""
Phase 6 — deep-learning semantic core contracts.

These tests lock the compiler architecture around first-class numerics,
state/cache, movement, typed collectives, schedule artifacts, and deterministic
effect semantics.
"""

from tessera.compiler.effects import Effect, EffectLattice
from tessera.compiler.graph_ir import KVCacheSpec, NumericPolicy


def test_numeric_policy_serializes_full_contract():
    p = NumericPolicy(
        storage="fp8_e4m3",
        accum="f32",
        rounding="stochastic",
        scale=0.5,
        quant_axis="per_channel_0",
        deterministic=True,
    )
    attr = p.to_mlir_attr()
    assert "storage = \"fp8_e4m3\"" in attr
    assert "accum = \"f32\"" in attr
    assert "rounding = \"stochastic\"" in attr
    assert "deterministic = true" in attr


def test_kv_cache_spec_serializes_as_state_object():
    spec = KVCacheSpec(max_seq=4096, head_dim=128)
    attrs = spec.create_attrs()
    assert "max_seq = 4096" in attrs
    assert "head_dim = 128" in attrs
    assert "numeric_policy" in attrs


def test_effect_lattice_distinguishes_movement_state_collective():
    lattice = EffectLattice()
    joined = lattice.join([Effect.movement, Effect.state, Effect.collective])
    assert joined == Effect.collective


def test_deterministic_allows_ir_visible_state_and_collectives():
    def step(cache, x):
        cache = ops.kv_cache_append(cache, x, x)  # noqa: F821
        y = ops.all_reduce(x)  # noqa: F821
        return y

    EffectLattice().check_deterministic(step, seed=42)
