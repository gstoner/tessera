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

    def test_audit_reports_fused_for_every_aliased_op(self) -> None:
        """Every M7 public op with a fused backend kernel must read as
        ``tile_ir=fused`` and ``target_ir=fused`` in the support table.

        The set of fused public ops is **derived** from
        ``audit.m7_fused_public_ops()`` rather than hardcoded — a new
        fused complex kernel widens this guard automatically and the
        matching guard in
        ``tests/unit/test_compiler_audit.py::test_visual_complex_rows_match_public_api_and_backend_aliases``
        widens at the same time (single source of truth).

        Floor invariant: the helper must today return at least
        ``{complex_mul, complex_exp, mobius, stereographic}`` —
        i.e., the four fused complex kernels that landed in
        ``_COMPLEX_APPLE_GPU_FUSED`` as of the M7 milestone (2026-05-19).
        Any regression that drops one will both fail this test and
        leave the helper undercounting.
        """

        from tessera.compiler.audit import (
            m7_fused_public_ops,
            support_row_for,
        )

        fused_ops = m7_fused_public_ops()
        # Floor invariant — the four M7 ops that have native MSL
        # kernels today must all be present.  Adding rows to
        # _COMPLEX_APPLE_GPU_FUSED widens the set further;
        # never let it shrink below this floor without an explicit
        # baseline update + a paired test edit.
        floor = {"complex_mul", "complex_exp", "mobius", "stereographic"}
        assert floor.issubset(fused_ops), (
            f"m7_fused_public_ops() dropped below the floor:\n"
            f"  expected ⊇ {sorted(floor)}\n"
            f"  got       {sorted(fused_ops)}"
        )

        for op_name in sorted(fused_ops):
            row = support_row_for(op_name)
            assert row.cells["tile_ir"].status == "fused", (
                f"{op_name}: expected tile_ir='fused', "
                f"got {row.cells['tile_ir'].status!r}"
            )
            assert row.cells["target_ir"].status == "fused", (
                f"{op_name}: expected target_ir='fused', "
                f"got {row.cells['target_ir'].status!r}"
            )

    def test_non_aliased_m7_ops_report_terminal_not_applicable(self) -> None:
        """E3 (2026-05-20): the M7 long-tail ops have no fused MSL
        kernel today, but they're not overclaimed as ``fused`` either.
        For the single-GPU closeout they read as
        ``target_ir=not_applicable`` — an intentional reference/domain
        composition, while apple_gpu / nvidia_sm90+ / rocm keep
        ``planned`` manifest slots reserved for the M7 kernel follow-up.
        """

        from tessera.compiler.audit import support_row_for
        from tessera.compiler.backend_manifest import manifest_for

        expected_rocm = {
            "cross_ratio": "planned",
            "dz": "compiled",
            "laplacian_2d": "compiled",
        }
        for op_name in ("cross_ratio", "dz", "laplacian_2d"):
            row = support_row_for(op_name)
            assert row.cells["tile_ir"].status == "not_applicable", (
                f"{op_name} reports tile_ir="
                f"{row.cells['tile_ir'].status!r}; expected "
                "'not_applicable' for the single-GPU reference/domain "
                "composition terminal state."
            )
            assert row.cells["target_ir"].status == "not_applicable", (
                f"{op_name} reports target_ir="
                f"{row.cells['target_ir'].status!r}; expected "
                f"'not_applicable' (the M7 long-tail runs via Python "
                f"reference/domain composition; native-kernel slots are "
                f"reserved as planned on the GPU targets)."
            )
            # The manifest must carry planned slots for the GPU
            # targets so Phase G / H / M7-follow-up knows where to
            # land the actual kernels.
            entries = {e.target: e.status for e in manifest_for(op_name)}
            for gpu in ("apple_gpu", "nvidia_sm90", "rocm"):
                expected = expected_rocm[op_name] if gpu == "rocm" else "planned"
                assert entries.get(gpu) == expected, (
                    f"{op_name}: expected {gpu}={expected}, got "
                    f"{entries.get(gpu)!r} — every M7 long-tail op "
                    f"should reserve a native-kernel slot for that "
                    f"target."
                )
