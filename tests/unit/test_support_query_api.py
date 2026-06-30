"""Unit tests for ``tessera.compiler.support`` (P0-1).

Locks the public query API:

  * :func:`support(op)` returns the same 8-axis data the audit
    table renders.
  * :func:`tier(op)` is the best-tier-across-targets rollup.
  * :func:`tier(op, target=...)` is the per-target view.
  * :func:`is_compiler_supported(op, target=...)` is True for
    artifact/reference/native compiler support, including
    hardware-gated targets.
  * :func:`is_native_supported(op, target=...)` is True iff the op
    has a fused kernel + ready runtime on that target.

The data layer is the same one ``support_table.md`` drift-gates,
so this test file doesn't duplicate the manifest — it just exercises
the query surface for a hand-picked set of ops covering each Tier.
"""

from __future__ import annotations

import pytest

from tessera.compiler import (
    OpSupport,
    TargetSupport,
    Tier,
    is_compiler_supported,
    is_native_supported,
    known_targets,
    support,
    tier,
)


class TestTierEnum:
    def test_tier_values_are_stable_strings(self) -> None:
        assert Tier.NATIVE_READY.value == "native_ready"
        assert Tier.REFERENCE_ONLY.value == "reference_only"
        assert Tier.ARTIFACT_ONLY.value == "artifact_only"
        assert Tier.PLANNED.value == "planned"

    def test_best_of_picks_lowest_rank(self) -> None:
        tiers = [Tier.PLANNED, Tier.NATIVE_READY, Tier.REFERENCE_ONLY]
        assert Tier.best_of(tiers) is Tier.NATIVE_READY

    def test_best_of_empty_defaults_to_planned(self) -> None:
        assert Tier.best_of([]) is Tier.PLANNED

    def test_is_at_least_native_dominates(self) -> None:
        assert Tier.NATIVE_READY.is_at_least(Tier.REFERENCE_ONLY)
        assert Tier.NATIVE_READY.is_at_least(Tier.NATIVE_READY)
        assert not Tier.REFERENCE_ONLY.is_at_least(Tier.NATIVE_READY)


class TestSupportShape:
    def test_returns_op_support_dataclass(self) -> None:
        info = support("matmul")
        assert isinstance(info, OpSupport)
        assert info.op_name == "matmul"
        # Every 8-axis field must be a non-empty string.
        for axis in (
            "api", "frontend", "graph_ir", "schedule_ir",
            "tile_ir", "target_ir", "runtime", "bench",
        ):
            value = getattr(info, axis)
            assert isinstance(value, str), axis
            assert value, axis

    def test_targets_is_non_empty(self) -> None:
        info = support("matmul")
        assert len(info.targets) > 0
        assert all(isinstance(ts, TargetSupport) for ts in info.targets)

    def test_for_target_returns_matching_row(self) -> None:
        info = support("matmul")
        ts = info.for_target("apple_gpu")
        assert ts.target == "apple_gpu"

    def test_for_target_accepts_target_aliases(self) -> None:
        info = support("matmul")
        assert info.for_target("apple").target == "apple_gpu"
        assert info.for_target("macos_gpu").target == "apple_gpu"

    def test_for_target_raises_on_unknown_target(self) -> None:
        info = support("matmul")
        with pytest.raises(KeyError):
            info.for_target("not_a_real_target")

    def test_as_dict_round_trips_to_plain_types(self) -> None:
        info = support("matmul")
        d = info.as_dict()
        assert d["op_name"] == "matmul"
        assert d["best_tier"] in {t.value for t in Tier}
        assert isinstance(d["axes"], dict)
        assert isinstance(d["targets"], list)
        assert all(isinstance(t, dict) for t in d["targets"])

    def test_known_targets_matches_capability_registry(self) -> None:
        from tessera.compiler.capabilities import TARGET_CAPABILITIES

        assert set(known_targets()) == set(TARGET_CAPABILITIES)


class TestTierResolution:
    """Spot-check tier values across the four tier categories.

    Each test pins one op as a canonical example.  When a new fused
    kernel lands, the corresponding op should still report the same
    tier — drop the test only if the op's status genuinely
    regresses.
    """

    def test_matmul_is_native_on_apple_gpu(self) -> None:
        assert tier("matmul", target="apple_gpu") is Tier.NATIVE_READY

    def test_matmul_is_reference_on_cpu(self) -> None:
        # CPU has fused/reference for matmul in the manifest;
        # the audit reports `reference` at target_ir today.
        assert tier("matmul", target="cpu") is Tier.REFERENCE_ONLY

    def test_matmul_best_tier_is_native(self) -> None:
        assert tier("matmul") is Tier.NATIVE_READY

    def test_m7_aliased_op_is_native_on_apple_gpu(self) -> None:
        # M7 alias check: `mobius` has a fused MSL kernel under the
        # `complex_mobius` name in the backend manifest.  The query
        # API must reflect that.
        assert tier("mobius", target="apple_gpu") is Tier.NATIVE_READY
        assert tier("stereographic", target="apple_gpu") is Tier.NATIVE_READY

    def test_m7_non_fused_op_is_planned(self) -> None:
        # `cross_ratio` is an M7 primitive with no fused kernel —
        # should report PLANNED at best.  Treat as floor invariant:
        # if a fused kernel lands, this test moves up rather than
        # disappearing.
        assert tier("cross_ratio") in (Tier.PLANNED, Tier.REFERENCE_ONLY)


class TestIsNativeSupported:
    def test_matmul_apple_gpu_true(self) -> None:
        assert is_native_supported("matmul", target="apple_gpu") is True

    def test_matmul_cpu_false(self) -> None:
        # Reference-only at CPU is not "native"
        assert is_native_supported("matmul", target="cpu") is False

    def test_unknown_target_raises(self) -> None:
        with pytest.raises(KeyError):
            is_native_supported("matmul", target="not_a_real_target")

    def test_planned_op_is_never_native(self) -> None:
        # cross_ratio has no fused kernel anywhere; should be False
        # on every known target.
        for target in known_targets():
            assert is_native_supported("cross_ratio", target=target) is False, target


class TestIsCompilerSupported:
    def test_artifact_only_backend_counts_as_compiler_supported(self) -> None:
        assert tier("matmul", target="nvidia_sm90") is Tier.ARTIFACT_ONLY
        assert is_compiler_supported("matmul", target="nvidia_sm90") is True
        assert is_native_supported("matmul", target="nvidia_sm90") is False

    def test_alias_target_counts_as_compiler_supported(self) -> None:
        assert is_compiler_supported("matmul", target="apple") is True

    def test_unknown_target_raises(self) -> None:
        with pytest.raises(KeyError):
            is_compiler_supported("matmul", target="not_a_real_target")


class TestNoParallelRegistryDrift:
    """The whole point of this module is to be a *derived* surface
    over the audit table.  This test asserts the derivation is
    correct on a per-axis basis for a sample op."""

    def test_support_axes_match_audit_row(self) -> None:
        from tessera.compiler.audit import support_row_for

        info = support("matmul")
        row = support_row_for("matmul")
        for axis in (
            "api", "frontend", "graph_ir", "schedule_ir",
            "tile_ir", "target_ir", "runtime", "bench",
        ):
            audit_status = row.cells[axis].status
            wrapper_status = getattr(info, axis)
            assert wrapper_status == audit_status, (
                f"{axis}: wrapper={wrapper_status!r} "
                f"audit={audit_status!r}"
            )

    def test_per_target_rows_use_alias_map(self) -> None:
        """``mobius`` must look up ``complex_mobius`` in the backend
        manifest via ``_M7_BACKEND_ALIASES`` — the wrapper's
        per-target view must agree with the audit's."""

        info = support("mobius")
        apple_gpu_row = info.for_target("apple_gpu")
        # Per audit + backend_manifest._COMPLEX_APPLE_GPU_FUSED
        assert apple_gpu_row.target_ir == "fused"
        assert apple_gpu_row.runtime == "fused"
