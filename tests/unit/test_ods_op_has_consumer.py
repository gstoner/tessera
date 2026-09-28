"""Host-free: Decision #29's op-level clause (GOV-ODS-CONSUMER-1).

#29 says a declared ODS op must have a named consumer or be deleted, and cites
`test_governance_declarations.py` as its drift gate -- which checks coverage
*axes* and duplicate *dialect names*, and never mapped an op to a consumer.
Confirmed the expensive way: `tessera.scaled_matmul` was committed with no
consumer and every governance test passed.

The scan lives in `tessera.compiler.ods_consumer_audit`; read its docstring for
what counts as a reference. The short version:

* a reference by **shipped compiler code** (a pass, lowering, verifier, or a
  Python producer/consumer) satisfies #29;
* a reference **only by tests or lit fixtures does not** -- a fixture proves the
  op parses, not that the compiler produces or consumes it -- so a
  fixture-only op must be waived and is listed here, never passed;
* an op nothing names is waived as `unreferenced`.

History: the first version of this gate (2026-09-20) matched only
`def X : Op<Dialect, "name">`, so it parsed 286 of 623 records and skipped the
dominant `def X : Dialect_Op<"name">` form entirely; it also counted fixtures
and bare substrings as consumers. Rebuilt 2026-09-27 (sync
`EVIDENCE-GOVERNANCE-GATES-2026-09-27`): a balanced TableGen reader that
resolves class templates, cross-checked record-for-record against
`llvm-tblgen --dump-json` once, by hand, when it landed (609 of 609 op records
agreed on name and mnemonic; the 14 it could not dump -- `tessera_neighbors.td`
redefined `StrAttr` (that file is deleted since, SMALL-CORRECTNESS-GAPS-2026-09-27),
and two solver `.td` files leave an attr/type `mnemonic` unresolved -- are read
here). No test re-runs that cross-check, since it needs
an LLVM install; `_DECLARED_OP_RECORDS` and the per-form pins stand in for it.
The pre-PR review found three fail-open holes (a `using` declaration, prose in a
string, an op's own dialect arity table); each has a synthetic case below.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

from tessera.compiler.ods_consumer_audit import (
    REPO_ROOT,
    OdsParseError,
    build_corpus,
    classify,
    declared_ops,
    duplicate_names,
    hand_declared_op_names,
)

_SELF = Path(__file__).resolve()

#: The waiver file names every waived op, so it is excluded from the corpus:
#: otherwise writing an op into the waiver would make it look referenced.
_OPS = declared_ops()
_TIERS = classify(_OPS, build_corpus(exclude=[_SELF], names={op.full_name for op in _OPS}))
_BY_KEY = {op.key: op for op in _OPS}


class Waiver(NamedTuple):
    """One declared op held without a compiler consumer.

    ``decision`` is ``"#29"`` for a plain violation (unfinished wiring or an op
    that should go -- the owner decides which) or ``"#29a"`` for a declared
    debt, which must also name ``owner``: a queue item that exists with a gate,
    and whose ID is written at the op's declaration site (#29a conditions 1-2;
    this file is condition 4, the count).
    """

    tier: str
    decision: str
    reason: str
    owner: str | None = None


# Per-op connection triage (capability, where it should connect, and a WIRE /
# #29a / merge / delete-candidate recommendation for every entry below):
# docs/audit/compiler/ODS_OP_CONNECTION_TRIAGE.md.
#
# Family reasons, measured 2026-09-27. "No static reference" leaves one door
# open that no scan can close: a name assembled at run time
# (`f"tessera.{kind}"`) is invisible here, so confirm before deleting.
_R_CACHE = ("cache dialect KV/page/ring ops: no pass, lowering, verifier or Python "
            "emitter names them; the KV cache lowers through `tessera.kv_cache.*` "
            "and runtime handles instead")
_R_TESSERA_CACHE = "Graph-level page lookup with no producer or lowering"
_R_SOLVER_CORE = ("solvers/core dialect (`trng`/`tsl`/`tss`): CMake runs TableGen "
                  "only (an INTERFACE library nothing compiles or registers), so no "
                  "tool can parse these ops; no C++ or Python names them")
_R_SOLVER_LINALG = ("linalg solver op with no producer or lowering; `potrf`/`potrs` "
                    "appear only in the `spd_solve.mlir` fixture")
_R_COLLECTIVE = ("collective dialect op with no producer or lowering in the "
                 "collective passes or the Python collectives surface")
_R_SCHEDULE = "Schedule IR op with no producer and no lowering"
_R_ARCH = ("architecture-search (`tessera.arch.*`) Graph op; neither the Python "
           "`tessera.arch` surface nor any pass names it statically")
_R_AMX = ("retired AMX ISA contract (Decision #19: AMX is a dead end, superseded by "
          "ACE); kept as an IR contract with no producer and no `amx.*` lowering")
_R_X86_DIRECTIVE = ("x86 Target IR directive with no producer: `TileToX86Pass` lowers to "
                    "`func.call` on the C shim and the Python x86 emitter emits "
                    "`tessera_x86.kernel`; named only by `x86_target_ir.mlir`")
_R_NVIDIA = ("sm_120 differentiation Target IR op named only by "
             "`sm120_differentiation_target_ir.mlir`; no C++ or Python producer")
_R_EBM_GRAPH = ("Graph-level EBM op: the Python `tessera.ebm` surface and the "
                "runtime kernels exist, but no frontend emits this op and no pass "
                "lowers it; named only by the `ga_ebm_graph_ops` fixtures")
_R_ATTN_RES = ("block-AttnRes state op: the Python `_block_attnres_ops` reference "
               "exists, but no frontend emits this op; only `depth_attn_verifier` "
               "fixtures name it")
_R_FIXTURE_ONLY = "named only by lit fixtures / unit tests; no compiler producer or consumer"
_R_UNREFERENCED = "nothing outside its own .td names it"
_R_CATALOG_ONLY = ("Graph op named by the op catalog and by tests; the scan does not "
                   "count the catalog, but `graph_ir._try_map_call` emits catalog "
                   "names from @jit bodies, so the frontend PRODUCES it -- what is "
                   "missing is a lowering consumer")
_R_CLIFFORD_CALCULUS = ("geometric-calculus op (derivative/integral) with no "
                        "producer and no lowering in the Clifford passes")
_R_ATTN_MASK = "FA-4 Attn Tile IR mask/LSE op with no producer; FA-4 lowering does not emit it"
_R_MOE = ("programming-model MoE op with no producer or lowering; its dialect is "
          "TableGen-only (never compiled or registered) and its fixture's RUN pass "
          "does not exist; the MoE transport tests exercise the Python API and the "
          "Graph `tessera.moe_dispatch`, not this op")


#: Shrink-only (ceiling below). Seeded 2026-09-27 from the first scan that
#: parsed every op; the 2026-09-20 list (20 names) was a subset of this, since
#: it parsed a third of the ops and passed fixture-only ones. **None of these
#: meets Decision #29a today** -- none is marked at its site with an owning
#: item -- so every entry is a plain #29 violation: wiring that was never
#: finished, or an op that should go. This PR deletes none of them; which
#: answer applies needs the person who owns each dialect.
_WAIVED: dict[str, Waiver] = {
    # cache dialect
    **{name: Waiver("unreferenced", "#29", _R_CACHE) for name in (
        "cache.kv.create", "cache.page.lookup", "cache.page.read", "cache.page.write",
        "cache.pt.create", "cache.ring.create", "cache.ring.pop", "cache.ring.push")},
    "tessera.cache.page_lookup": Waiver("unreferenced", "#29", _R_TESSERA_CACHE),
    # solver dialects
    **{name: Waiver("unreferenced", "#29", _R_SOLVER_CORE) for name in (
        "trng.create_state", "trng.uniform", "trng.normal", "tsl.root.brent",
        "tsl.root.newton", "tsl.solve_trig", "tss.spmv", "tss.spmm", "tss.cg", "tss.gmres")},
    # getrf/potrf/potrs/trsm left this list 2026-09-27
    # (TILE-LATENT-DEFECTS-2026-09-27): the linalg MixedPrecision /
    # IterativeRefinement passes now select them by op identity instead of a
    # `contains("solve")` substring. Those passes only annotate (no pass reads
    # `tessera.compute_dtype` / `tessera_solver.ir_*` yet), which is an
    # attribute-level #29 gap, not an op-level one.
    "tessera_solver.ir_step": Waiver("unreferenced", "#29", _R_SOLVER_LINALG),
    "tessera_sr.export_manifest": Waiver("unreferenced", "#29", _R_UNREFERENCED),
    "tessera_spectral.twiddle_table": Waiver("unreferenced", "#29", _R_UNREFERENCED),
    # collectives
    **{name: Waiver("unreferenced", "#29", _R_COLLECTIVE) for name in (
        "tessera_collective.pack_cast", "tessera_collective.shard_view",
        "tessera_collective.materialize_shard", "tessera_collective.qos.acquire",
        "tessera_collective.qos.release")},
    "tessera_collective.qos.limit": Waiver("fixture_only", "#29", _R_COLLECTIVE),
    # schedule / programming model
    **{name: Waiver("unreferenced", "#29", _R_SCHEDULE) for name in (
        "schedule.optimizer_shard", "schedule.async_copy", "schedule.await_movement")},
    **{name: Waiver("fixture_only", "#29", _R_MOE) for name in (
        "moe.plan", "moe.token_limiter.create")},
    "moe.dispatch": Waiver("unreferenced", "#29", _R_MOE),
    # Graph IR
    **{name: Waiver("unreferenced", "#29", _R_ARCH) for name in (
        "tessera.arch.weighted_sum", "tessera.arch.switch", "tessera.arch.mixed")},
    "tessera.arch.parameter": Waiver("fixture_only", "#29", _R_ARCH),
    **{name: Waiver("fixture_only", "#29", _R_EBM_GRAPH + "; the frontend emits the "
                    "flat spelling (`tessera.ebm_inner_step` / `ebm_self_verify`) the "
                    "runtime consumes instead (triage: merge/supersede)") for name in (
        "tessera.ebm.inner_step", "tessera.ebm.self_verify")},
    "tessera.ebm.decode_init": Waiver("fixture_only", "#29", _R_EBM_GRAPH),
    **{name: Waiver("fixture_only", "#29", _R_EBM_GRAPH + "; the compiled capability "
                    "is `tessera_ebm.langevin_step{manifold}` (native_langevin.py -> "
                    "LowerLangevin), triage: merge/supersede") for name in (
        "tessera.ebm.bivector_langevin_step", "tessera.ebm.sphere_langevin_step")},
    **{name: Waiver("fixture_only", "#29", _R_ATTN_RES) for name in (
        "tessera.attn_with_stats", "tessera.softmax_merge", "tessera.softmax_finalize")},
    "tessera.guided_denoise_region": Waiver("fixture_only", "#29", _R_FIXTURE_ONLY),
    "tessera.istft_jvp": Waiver(
        "fixture_only", "#29", "produced by `ISTFTOp::buildTangent` "
        "(TangentInterface.cpp) under --tessera-autodiff-forward, which the scan "
        "reads as the dialect's own implementation; nothing lowers it"),
    # Added 2026-09-27 by the review that closed three fail-open holes; each
    # was "consumed" only through one of them.
    "tessera.arch.ste_one_hot": Waiver(
        "unreferenced", "#29", _R_ARCH + "; its only mention was its own "
        "dialect's arity table in TesseraOps.cpp"),
    **{name: Waiver("fixture_only", "#29", _R_CATALOG_ONLY) for name in (
        "tessera.cache.commit", "tessera.cache.rollback", "tessera.ntk_rope",
        "tessera.target_verify")},
    "tessera.ebm.langevin_step_philox": Waiver(
        "fixture_only", "#29", _R_EBM_GRAPH + "; the runtime kernels mirror its "
        "semantics, and its only other mention was prose in an execution-matrix "
        "`reason=` string; the compiled executors run its Philox semantics under "
        "the name `tessera.ebm.langevin_step`"),
    "tessera_nvidia.func": Waiver(
        "fixture_only", "#29", "NVIDIA Target IR container op named only by "
        "fixtures; the bare `FuncOp` in PipelineOverlapPass.cpp is "
        "`using mlir::func::FuncOp`, not this op"),
    "tessera.ring.create": Waiver("fixture_only", "#29", _R_FIXTURE_ONLY),
    # Tile / Attn / domain dialects
    **{name: Waiver("fixture_only", "#29", _R_ATTN_MASK) for name in (
        "tessera_attn.lse.save", "tessera_attn.lse.load", "tessera_attn.causal_mask")},
    "tessera_attn.dropout_mask": Waiver("unreferenced", "#29", _R_ATTN_MASK),
    **{name: Waiver("unreferenced", "#29", _R_CLIFFORD_CALCULUS) for name in (
        "tessera_clifford.ext_deriv", "tessera_clifford.codiff",
        "tessera_clifford.vec_deriv", "tessera_clifford.integral")},
    "tessera_ebm.partition_z": Waiver("fixture_only", "#29", _R_FIXTURE_ONLY),
    # Target IR
    "tessera_apple.gpu.mps_softmax": Waiver("unreferenced", "#29", _R_UNREFERENCED),
    **{name: Waiver("unreferenced", "#29", _R_UNREFERENCED) for name in (
        "tessera_rocm.memcpy", "tessera_rocm.emit")},
    **{name: Waiver("fixture_only", "#29", _R_NVIDIA) for name in (
        "tessera_nvidia.mma_fused", "tessera_nvidia.mma_attention", "tessera_nvidia.fpquant")},
    **{name: Waiver("fixture_only", "#29", _R_AMX) for name in (
        "tessera_x86.amx_tile_load", "tessera_x86.amx_tile_store",
        "tessera_x86.amx_tile_zero", "tessera_x86.amx_dpbf16ps")},
    "tessera_x86.amx_dpbusd": Waiver("unreferenced", "#29", _R_AMX),
    **{name: Waiver("fixture_only", "#29", _R_X86_DIRECTIVE) for name in (
        "tessera_x86.avx512_gemm_microkernel", "tessera_x86.pack_b_panel",
        "tessera_x86.elementwise")},
}

#: The waiver may only shrink: lower this with every entry removed. Raising it
#: is visible in review and needs a reason in the PR.
_WAIVER_CEILING = 79

#: History (the ratchet this replaced): on 2026-09-27 seven `tessera.neighbors.*`
#: names were declared by two ODS records -- `TesseraOps.td` (the live ones:
#: MLIR resolves `tessera.neighbors.x` by its first segment) and an unbuilt
#: `tessera_neighbors.td` -- and a third time by hand in `TesseraNeighbors.cpp`.
#: Consolidated the same day onto `TesseraOps.td` (sync
#: `SMALL-CORRECTNESS-GAPS-2026-09-27`), so the baseline is now empty and both
#: gates below are absolute: no op name may have a second declaration of any form.


# ─── The scan itself ────────────────────────────────────────────────────────


def test_scan_parses_every_ods_form() -> None:
    """A scan that silently skips a declaration form passes every other test.

    Pins one op per form: the class-template form that the 2026-09-20 regex
    missed, a two-level class chain, the direct `Op<Dialect, "name">` form, a
    dotted dialect name, and a dotted mnemonic.
    """
    names = {op.full_name for op in _OPS}
    for expected in (
        "tile.view",                       # def Tile_ViewOp : Tile_Op<"view", [Pure]>
        "tile.tri_solve",                  # Tile_LinalgOp -> Tile_Op -> Op
        "tessera_solver.trsm",             # def TrsmOp : Op<Tessera_Solver_Dialect, "trsm">
        "tessera.matmul",
        "tessera_rocm.swmmac",
        "tile.tmem.store",                 # dotted mnemonic
        "tessera.neighbors.halo.region",   # dotted dialect name
    ):
        assert expected in names, f"the ODS reader no longer parses {expected}"
    assert len(_OPS) == _DECLARED_OP_RECORDS, (
        f"{len(_OPS)} ODS op records parsed, expected {_DECLARED_OP_RECORDS}. "
        f"If you added or deleted ops, update _DECLARED_OP_RECORDS; otherwise the "
        f"reader has drifted from the ODS spelling and ops are escaping the gate "
        f"(or being invented).")


#: Every op record under `src/`. Pinned exactly, not as a floor: a floor let
#: the reader lose up to its slack without failing (GOV-ODS-CONSUMER-1 review).
_DECLARED_OP_RECORDS = 616


def test_scan_calls_a_known_consumed_op_consumed() -> None:
    assert _TIERS["tessera.matmul"] == "compiler"
    assert _TIERS["tessera_rocm.swmmac"] == "compiler"


def _synthetic_repo(tmp: Path) -> None:
    (tmp / "src/d").mkdir(parents=True)
    (tmp / "src/lib").mkdir(parents=True)
    (tmp / "tests").mkdir()
    (tmp / "src/d/Foo.td").write_text(textwrap.dedent('''
        // def Commented_Op : Op<Foo_Dialect, "ghost">;
        def Foo_Dialect : Dialect {
          let name = "foo";
          let cppNamespace = "::acme::foo";
        }
        class Foo_Op<string mnemonic, list<Trait> traits = []> :
            Op<Foo_Dialect, mnemonic, traits>;
        class Foo_PureOp<string m> : Foo_Op<m, [Pure]> {
          let summary = [{ a code block naming foo.unused, which is not a use }];
        }
        def Foo_UsedOp : Foo_Op<"used"> {}
        def Foo_TextualOp : Foo_PureOp<"textual.op">;
        def Foo_FixtureOp : Foo_Op<"fixture"> {}
        def Foo_UnusedOp : Op<Foo_Dialect, "unused", [Pure]> {}
        def Foo_YieldOp : Foo_Op<"yield"> {}
        def Foo_FuncOp : Foo_Op<"func"> {}
        def Foo_ProseOp : Foo_Op<"prose"> {}
        def Foo_TableOp : Foo_Op<"table"> {}
        def Foo_CatalogOp : Foo_Op<"catalog"> {}
        def Foo_WildOp : Foo_Op<"wild"> {}
    '''))
    (tmp / "src/lib/Lower.cpp").write_text(textwrap.dedent('''
        // UnusedOp is mentioned in a comment, which is not a use.
        LogicalResult UnusedOp::verify() { return success(); }
        struct P : OpRewritePattern<UsedOp> {};
        void f() {
          auto s = "%0 = foo.textual.op";
          auto why = "this lowering matches foo.prose.";  // prose, not IR
          scf::YieldOp y;           // a foreign YieldOp, not foo.yield
          auto t = x.foo.unused;    // member access, not a textual op name
        }
    '''))
    (tmp / "src/lib/Other.cpp").write_text(textwrap.dedent('''
        using mlir::func::FuncOp;
        void g(FuncOp f) {}         // mlir's FuncOp, not foo.func
    '''))
    (tmp / "src/lib/Upstream.cpp").write_text(textwrap.dedent('''
        using namespace mlir::scf;
        void h(WildOp w) {}         // could be scf's; not evidence for foo.wild
    '''))
    # The dialect's own implementation: its arity table is not a consumer.
    (tmp / "src/lib/FooOps.cpp").write_text(textwrap.dedent('''
        static const Arity kTable[] = {{"foo.table", 1, 1}};
        LogicalResult TableOp::verify() { return success(); }
        #define GET_OP_CLASSES
        #include "FooOps.cpp.inc"
    '''))
    (tmp / "python/tessera/compiler").mkdir(parents=True)
    (tmp / "python/tessera/compiler/op_catalog.py").write_text('OPS = ["foo.catalog"]\n')
    (tmp / "tests/fixture.mlir").write_text("%0 = foo.fixture : i32\n")


def test_scan_on_a_synthetic_dialect(tmp_path: Path) -> None:
    """Every rule in one place: tier, template resolution, and each non-use."""
    _synthetic_repo(tmp_path)
    ops = declared_ops(tmp_path)
    assert sorted(op.full_name for op in ops) == [
        "foo.catalog", "foo.fixture", "foo.func", "foo.prose", "foo.table",
        "foo.textual.op", "foo.unused", "foo.used", "foo.wild", "foo.yield"]
    tiers = classify(ops, build_corpus(tmp_path))
    assert tiers == {
        "foo.used": "compiler",            # OpRewritePattern<UsedOp>
        "foo.textual.op": "compiler",      # textual name used as IR in a string
        "foo.fixture": "fixture_only",     # a lit fixture alone is not a consumer
        "foo.unused": "unreferenced",      # own verify(), a comment, member access
        "foo.yield": "unreferenced",       # scf::YieldOp is not foo's YieldOp
        "foo.func": "unreferenced",        # `using mlir::func::FuncOp`
        "foo.wild": "unreferenced",        # bare name under an upstream using-namespace
        "foo.prose": "unreferenced",       # a sentence mentioning it
        "foo.table": "unreferenced",       # its own dialect's arity table
        "foo.catalog": "unreferenced",     # a registry lists names; it consumes none
    }


@pytest.mark.parametrize("td_text, match", [
    ('def D : Dialect { let name = "d"; }\ndefm X : Many<"a">;\n', "defm"),
    ('def D : Dialect { let name = "d"; }\ndef X : Unknown_Op<"a">;\n', "not declared"),
    ('def D : Dialect { let name = "d"; }\nclass C<string m> : Op<D, m>;\n'
     'class C<string m> : Op<D, m>;\n', "declared twice"),
])
def test_the_reader_refuses_what_it_cannot_read(tmp_path: Path, td_text: str, match: str) -> None:
    """An op the reader cannot resolve would silently leave the gate."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/X.td").write_text(td_text)
    with pytest.raises(OdsParseError, match=match):
        declared_ops(tmp_path)


def test_unparseable_python_is_an_error_not_a_blanket_reference(tmp_path: Path) -> None:
    _synthetic_repo(tmp_path)
    (tmp_path / "python/tessera/compiler/broken.py").write_text('x = "foo.unused"(\n')
    with pytest.raises(OdsParseError, match="cannot parse"):
        build_corpus(tmp_path, names={"foo.unused"})


# ─── The gate ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(_TIERS), ids=sorted(_TIERS))
def test_declared_op_has_a_compiler_consumer(key: str) -> None:
    tier = _TIERS[key]
    if tier == "compiler":
        return
    op = _BY_KEY[key]
    where = op.td.relative_to(REPO_ROOT)
    assert key in _WAIVED, (
        f"{key} ({op.record}, {where}) is {tier.replace('_', '-')}: no pass, "
        f"lowering, verifier or Python producer/consumer names it"
        + (" -- only tests do, and a fixture proves the op parses, not that "
           "the compiler uses it" if tier == "fixture_only" else "")
        + ". Decision #29: give it a consumer or delete it. If it is deliberate "
          "debt, Decision #29a requires it be marked AT THE SITE with the "
          "owning queue item before it is added to _WAIVED.")
    assert _WAIVED[key].tier == tier, (
        f"_WAIVED[{key!r}] says {_WAIVED[key].tier} but the scan measures {tier}; "
        f"update the entry so the waiver states the current fact")


def test_waiver_only_shrinks() -> None:
    stale = []
    for key in sorted(_WAIVED):
        if key not in _TIERS:
            stale.append(f"{key} (no longer declared)")
        elif _TIERS[key] == "compiler":
            stale.append(f"{key} (now has a compiler consumer)")
    assert not stale, f"remove from _WAIVED: {stale}"
    assert len(_WAIVED) <= _WAIVER_CEILING, (
        f"_WAIVED grew to {len(_WAIVED)} past its ceiling {_WAIVER_CEILING}; a "
        f"new op needs a consumer, not a waiver")
    assert len(_WAIVED) == _WAIVER_CEILING, (
        f"_WAIVED shrank to {len(_WAIVED)}: lower _WAIVER_CEILING to match so the "
        f"freed slot cannot be reused")


def _plan_has_item(item: str) -> bool:
    heading = re.compile(rf"^#+\s+.*\b{re.escape(item)}\b", re.M)
    plans = [REPO_ROOT / "docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md",
             *sorted((REPO_ROOT / "docs/audit/backend").glob("*/todo.md"))]
    return any(heading.search(p.read_text(errors="replace")) for p in plans if p.exists())


def _marked_at_site(td: Path, record: str, owner: str) -> bool:
    """The owner ID appears in the record's definition or its leading comment."""
    lines = td.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(record)}\b", line):
            start = i
            while start > 0 and lines[start - 1].lstrip().startswith("//"):
                start -= 1
            end = i
            while end < len(lines) and not lines[end].startswith("}") \
                    and not lines[end].rstrip().endswith(";"):
                end += 1
            return owner in "\n".join(lines[start:end + 1])
    return False


@pytest.mark.parametrize("key", sorted(_WAIVED))
def test_each_waiver_is_well_formed(key: str) -> None:
    waiver = _WAIVED[key]
    assert waiver.tier in ("fixture_only", "unreferenced")
    assert len(waiver.reason) >= 25, f"{key}: a waiver must say why"
    assert waiver.decision in ("#29", "#29a")
    if waiver.decision == "#29":
        assert waiver.owner is None, f"{key}: a plain #29 violation has no owning debt item"
        return
    # Decision #29a, conditions 1 and 2. (3, a behavioural test, is the
    # owner's; 4, being counted, is this list.)
    assert waiver.owner, f"{key}: a #29a debt must name its owning queue item"
    assert _plan_has_item(waiver.owner), (
        f"{key}: owner {waiver.owner} is not a queue item with a heading")
    op = _BY_KEY[key]
    assert _marked_at_site(op.td, op.record, waiver.owner), (
        f"{key}: #29a requires {op.record} be marked at its declaration with "
        f"{waiver.owner}; otherwise it is a plain #29 violation")


def test_marked_at_site_reads_the_definition(tmp_path: Path) -> None:
    """The #29a site check itself, so an unused branch cannot be vacuous."""
    td = tmp_path / "X.td"
    td.write_text("// Unwired: owned by GOV-ODS-CONSUMER-1.\n"
                  "def Foo_BarOp : Foo_Op<\"bar\"> {\n}\n"
                  "def Foo_BazOp : Foo_Op<\"baz\">;\n")
    assert _marked_at_site(td, "Foo_BarOp", "GOV-ODS-CONSUMER-1")
    assert not _marked_at_site(td, "Foo_BazOp", "GOV-ODS-CONSUMER-1")
    assert _plan_has_item("GOV-ODS-CONSUMER-1")
    assert not _plan_has_item("NO-SUCH-ITEM-9")


def test_no_two_records_declare_one_op_name() -> None:
    dups = duplicate_names(_OPS)
    assert not dups, (
        f"these op names are declared by more than one ODS record: {dups}. "
        f"MLIR resolves a name by its first segment, so at most one can be "
        f"live; delete the other declaration (Decision #31)")


def test_no_hand_written_cpp_op_shadows_an_ods_op() -> None:
    """The third form `duplicate_names` cannot see: a C++ `Op<>` class by hand.

    A hand-rolled op registering an ODS-declared name is a second authority
    whose verifier may never run (the deleted neighbors dialect's did not: the
    parser resolved every name to the `tessera` dialect) and which, loaded next
    to the ODS dialect, would register one name twice.
    """
    ods = {op.full_name for op in _OPS}
    hand = hand_declared_op_names()
    shadowed = {name: files for name, files in hand.items() if name in ods}
    assert not shadowed, (
        f"hand-written C++ ops re-declare ODS op names: {shadowed}. Keep the "
        f"ODS declaration and move any verifier logic into its verify().")


def test_hand_written_op_scan_sees_the_form_it_gates(tmp_path) -> None:
    """The scanner must match the shape the deleted neighbors dialect used."""
    src = tmp_path / "src" / "x"
    src.mkdir(parents=True)
    (src / "Hand.cpp").write_text(
        "struct HaloRegionOp : Op<HaloRegionOp> {\n"
        "  static llvm::StringRef getOperationName() {\n"
        "    return \"tessera.neighbors.halo.region\";\n  }\n};\n"
        "// static StringRef getOperationName() { return \"in.a.comment\"; }\n")
    (src / "Use.cpp").write_text(
        "auto n = HaloRegionOp::getOperationName();\n")
    found = hand_declared_op_names(tmp_path)
    assert found == {"tessera.neighbors.halo.region": ["src/x/Hand.cpp"]}
