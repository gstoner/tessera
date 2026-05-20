"""Regression guards for the M7 (Visual Complex Analysis) audit visibility fixes.

Locks the three findings from the 2026-05-19 review:

1. ``mobius`` and ``stereographic`` map to backend-manifest entries
   ``complex_mobius`` / ``complex_stereographic`` via
   ``_M7_BACKEND_ALIASES`` so the audit reflects fused-kernel coverage
   instead of underclaiming ``target_ir=planned``.
2. ``complex_add`` is NOT in ``_M7_INVENTORY`` — there is no
   ``def complex_add`` in ``python/tessera/complex.py``; adding the
   row was an overclaim.
3. Every M7 row in ``_M7_INVENTORY`` is backed by a real public
   function in ``tessera.complex`` OR carries an explicit alias to a
   public surface (``complex_jit`` lives in
   ``tessera.compiler.complex_jit``).
"""

from __future__ import annotations

import importlib

import pytest


class TestM7Inventory:
    def test_no_complex_add_overclaim(self) -> None:
        from tessera.compiler.audit import _M7_INVENTORY
        assert "complex_add" not in _M7_INVENTORY, (
            "complex_add must NOT appear in _M7_INVENTORY — no "
            "``def complex_add`` exists in python/tessera/complex.py; "
            "complex addition is just `+` on numpy arrays."
        )

    def test_every_m7_op_is_publicly_callable(self) -> None:
        """Every name in ``_M7_INVENTORY`` must resolve to a real
        callable in one of the documented public surfaces."""

        from tessera.compiler.audit import _M7_INVENTORY

        complex_mod = importlib.import_module("tessera.complex")
        complex_jit_mod = importlib.import_module(
            "tessera.compiler.complex_jit"
        )
        public_surfaces = (complex_mod, complex_jit_mod)

        missing: list[str] = []
        for name in sorted(_M7_INVENTORY):
            if any(hasattr(s, name) for s in public_surfaces):
                continue
            missing.append(name)
        assert missing == [], (
            "M7 inventory contains entries with no matching public "
            "function in tessera.complex or "
            "tessera.compiler.complex_jit. Either add the function "
            "and tests, or remove the row from _M7_INVENTORY:\n"
            + "\n".join(f"  - {n}" for n in missing)
        )


class TestM7BackendAlias:
    def test_alias_map_targets_real_manifest_entries(self) -> None:
        """Every aliased name must exist in
        ``backend_manifest._COMPLEX_APPLE_GPU_FUSED`` so the audit's
        lookup actually finds a fused entry."""

        from tessera.compiler.audit import _M7_BACKEND_ALIASES
        from tessera.compiler import backend_manifest as bm

        for public_name, backend_name in _M7_BACKEND_ALIASES.items():
            assert backend_name in bm._COMPLEX_APPLE_GPU_FUSED, (
                f"_M7_BACKEND_ALIASES maps {public_name!r} → "
                f"{backend_name!r}, but {backend_name!r} is not in "
                f"backend_manifest._COMPLEX_APPLE_GPU_FUSED."
            )

    @pytest.mark.parametrize(
        "op_name,expected_tile_ir,expected_target_ir",
        [
            ("mobius", "fused", "fused"),
            ("stereographic", "fused", "fused"),
            ("complex_mul", "fused", "fused"),
            ("complex_exp", "fused", "fused"),
        ],
    )
    def test_audit_reports_fused_for_aliased_ops(
        self,
        op_name: str,
        expected_tile_ir: str,
        expected_target_ir: str,
    ) -> None:
        from tessera.compiler.audit import support_row_for

        row = support_row_for(op_name)
        assert row.cells["tile_ir"].status == expected_tile_ir, (
            f"{op_name}: expected tile_ir={expected_tile_ir!r}, "
            f"got {row.cells['tile_ir'].status!r}"
        )
        assert row.cells["target_ir"].status == expected_target_ir, (
            f"{op_name}: expected target_ir={expected_target_ir!r}, "
            f"got {row.cells['target_ir'].status!r}"
        )

    def test_non_aliased_m7_ops_still_planned(self) -> None:
        """Sanity check the alias map is narrow — ops we did NOT
        alias still report target_ir=planned, so we're not over-
        promoting the rest of the M7 family."""

        from tessera.compiler.audit import support_row_for

        # `cross_ratio` and `dz` have no fused MSL kernel today;
        # they should still read as `planned` on the lowering axes.
        for op_name in ("cross_ratio", "dz", "laplacian_2d"):
            row = support_row_for(op_name)
            assert row.cells["target_ir"].status == "planned", (
                f"{op_name} unexpectedly reports "
                f"target_ir={row.cells['target_ir'].status!r}; the "
                f"alias map widened too far."
            )
