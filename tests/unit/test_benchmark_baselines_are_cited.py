"""Every sealed baseline is cited by something outside `benchmarks/baselines/`.

An evidence file nobody names is a claim nobody can check: it cannot be read
back through a dashboard, a log entry or a test, so it is either a stale
artifact or evidence for a record that no longer says where it came from. The
2026-09-17 review found 31 top-level baseline files and four packet directories
in that state, some dating to July, and ten packet directories with no manifest
or README at all.

Those orphans were pruned on 2026-09-17 (owner-approved): 22 of the 31 files
turned out to be read or cited by a *derived* name (a glob reader, a brace
expansion or a family glob in a record) and moved to `CITED_BY_DERIVED_NAME`;
the other nine files and all four directories were deleted; the ten
manifestless directories received a README naming their recorder and citer.
The ratchet now holds at an empty floor: a new baseline that lands uncited, a
new packet directory nothing cites, or one with no manifest/README fails here,
on the CPU-only unit lane. Never add an entry to the frozen sets to make a new
orphan pass; cite it from a record or do not seal it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "benchmarks" / "baselines"
CITING_ROOTS = ("tests", "docs", "python", "scripts", "benchmarks", "tools")

#: Orphans as of 2026-09-17, pruned the same day — empty by construction.
#: Never add an entry here to make a new orphan pass.
KNOWN_UNCITED_FILES: frozenset[str] = frozenset()
#: Baselines a textual whole-token search misses although a reader or a record
#: names them by a *derived* name. Each group names its reader.
CITED_BY_DERIVED_NAME = frozenset({
    # `apple_route_selector.legacy_route_ledger_inventory` rewrites
    # `*_route_ledger.json` to `*_strict_v2_route_ledger.json`.
    "apple7_attention_strict_v2_route_ledger.json",
    "apple7_epilogue_strict_v2_route_ledger.json",
    "apple7_gemm_strict_v2_route_ledger.json",
    "apple7_package_subgraph_strict_v2_route_ledger.json",
    # `benchmarks/nvidia/record_autotune_reproducibility.py::_resource_fingerprints`
    # loads every `benchmarks/baselines/nvidia*resource*.json` by glob.
    "nvidia_sm120_emitted_gemm_resources.json",
    "nvidia_sm120_replay_resources.json",
    "nvidia_sm120_test5_resources.json",
    "nvidia_sm120_transport_serving_resources.json",
    # `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` cites these by brace expansion:
    # `native_storage_pair_{nvidia,rocm,apple}.json`,
    # `native_storage_pending_ring{3,4,8}_nvidia.json`,
    # `native_storage_reduction_vjp_{nvidia,rocm,apple}.json`,
    # `native_storage_ring{3,4,8}_nvidia.json`.
    "native_storage_pair_apple.json",
    "native_storage_pair_nvidia.json",
    "native_storage_pair_rocm.json",
    "native_storage_pending_ring3_nvidia.json",
    "native_storage_pending_ring4_nvidia.json",
    "native_storage_pending_ring8_nvidia.json",
    "native_storage_reduction_vjp_apple.json",
    "native_storage_reduction_vjp_nvidia.json",
    "native_storage_reduction_vjp_rocm.json",
    "native_storage_ring3_nvidia.json",
    "native_storage_ring4_nvidia.json",
    "native_storage_ring8_nvidia.json",
    # `docs/audit/compiler/INTEGRATED_COMPILER_LOG.md` (native ring protocol
    # entry) cites these by brace expansion: `ring_protocol_{nvidia,rocm}.json`,
    # `ring_protocol_nvidia_ncu_{direct,depth2}.csv`,
    # `ring_protocol_nvidia_nsys_{kernels,api}.csv`.
    "ring_protocol_nvidia.json",
    "ring_protocol_nvidia_ncu_depth2.csv",
    "ring_protocol_nvidia_ncu_direct.csv",
    "ring_protocol_nvidia_nsys_api.csv",
    "ring_protocol_nvidia_nsys_kernels.csv",
    "ring_protocol_rocm.json",
    # `docs/audit/backend/x86/todo.md` cites the committed x86 baselines as the
    # `core_compiler_*_avx512.json` family (its route-ledger assessment).
    "core_compiler_training_backward_gfx1151_avx512.json",
    "core_compiler_training_step_fusion_gfx1151_avx512.json",
    "core_compiler_x86_layout_materialization_avx512.json",
})
KNOWN_UNCITED_DIRS: frozenset[str] = frozenset()
KNOWN_MANIFESTLESS_DIRS: frozenset[str] = frozenset()


def _cited_names(names: list[str]) -> set[str]:
    """Which of `names` appear as a whole identifier token in a tracked file
    under the citing roots. One pass, tokenised — the same method as
    `test_benchmark_recorders_are_named.py`; a `git grep -F` over ~200 names
    cost 10 s, this costs about two. This test file is excluded, or the frozen
    lists above would cite every orphan they name.
    """
    import re

    wanted = set(names)
    tracked = subprocess.run(
        ["git", "ls-files", "--", *CITING_ROOTS, ":!benchmarks/baselines"],
        cwd=ROOT, capture_output=True, text=True, check=False).stdout.split("\n")
    token = re.compile(r"[A-Za-z0-9_]+")
    self_name = f"tests/unit/{Path(__file__).name}"
    cited: set[str] = set()
    for rel in tracked:
        if not rel or rel == self_name:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cited |= set(token.findall(text)) & wanted
    return cited


def test_every_top_level_baseline_is_cited_or_frozen():
    files = [p for p in BASELINES.iterdir() if p.is_file() and p.name not in CITED_BY_DERIVED_NAME]
    cited = _cited_names([p.stem for p in files])
    uncited = sorted(p.name for p in files if p.stem not in cited)
    new = sorted(set(uncited) - KNOWN_UNCITED_FILES)
    assert not new, f"new uncited baseline file(s) — cite them from a record, or do not seal them: {new}"
    healed = sorted(KNOWN_UNCITED_FILES - set(uncited))
    assert not healed, f"these are now cited (or gone); remove them from KNOWN_UNCITED_FILES: {healed}"


def test_every_packet_directory_is_cited_or_frozen():
    dirs = [p for p in BASELINES.iterdir() if p.is_dir()]
    cited = _cited_names([d.name for d in dirs])
    uncited = sorted(d.name for d in dirs if d.name not in cited)
    new = sorted(set(uncited) - KNOWN_UNCITED_DIRS)
    assert not new, f"new uncited packet directory(ies): {new}"
    healed = sorted(KNOWN_UNCITED_DIRS - set(uncited))
    assert not healed, f"now cited (or gone); remove from KNOWN_UNCITED_DIRS: {healed}"


def test_every_packet_directory_has_a_manifest_or_is_frozen():
    dirs = [p for p in BASELINES.iterdir() if p.is_dir()]
    bare = sorted(d.name for d in dirs
                  if not any((d / n).exists() for n in ("manifest.json", "README.md", "index.json", "packet.json")))
    new = sorted(set(bare) - KNOWN_MANIFESTLESS_DIRS)
    assert not new, f"new packet directory(ies) with no manifest or README: {new}"
    healed = sorted(KNOWN_MANIFESTLESS_DIRS - set(bare))
    assert not healed, f"now have a manifest (or gone); remove from KNOWN_MANIFESTLESS_DIRS: {healed}"
