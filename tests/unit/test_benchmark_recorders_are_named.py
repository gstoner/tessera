"""Every benchmark recorder is named by some other tracked file.

A recorder nothing names — no README row, no packet manifest, no test, no log
entry — is a declaration without a consumer (Decision #29): it reads in the tree
as a lane that exists, while nothing can say what it recorded or where. The
2026-09-17 review found 50 such files, 21 of them `benchmarks/nvidia/record_*`
whose products are cited by date in the NVIDIA queue but never by the recorder
that made them.

A ratchet, not a claim of cleanliness: the 50 are frozen below and may only
shrink. A new recorder that lands unnamed fails here, on the CPU-only lane.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMING_ROOTS = ("tests", "docs", "python", "scripts", "benchmarks", "tools", ".github", "CLAUDE.md", "AGENTS.md")

#: Orphans as of 2026-09-17. Remove an entry once something names its stem;
#: never add one to make a new orphan pass.
KNOWN_UNNAMED = frozenset({
    "benchmarks/Tessera_Operator_Benchmarks/scripts/plot_csv.py",
    "benchmarks/Tessera_SuperBench/benches/kernel/conv2d_nhwc_tessera_stub.py",
    "benchmarks/Tessera_SuperBench/benches/kernel/flashattn_tessera_stub.py",
    "benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera_stub.py",
    "benchmarks/apple_gpu/benchmark_coopmat.py",
    "benchmarks/apple_gpu/benchmark_e2e2_gelu.py",
    "benchmarks/nvidia/benchmark_scheduled_macro_matmul.py",
    "benchmarks/nvidia/prepare_test5_profile_artifacts.py",
    "benchmarks/nvidia/profile_gemm_schedule_candidates.py",
    "benchmarks/nvidia/profile_test5_emitted_gemm.py",
    "benchmarks/nvidia/profile_test5_routes.py",
    "benchmarks/nvidia/record_attention_forward_schedule_matrix.py",
    "benchmarks/nvidia/record_autotune_reproducibility.py",
    "benchmarks/nvidia/record_bf16_reduction_breadth.py",
    "benchmarks/nvidia/record_canonical_k_loop.py",
    "benchmarks/nvidia/record_deltanet_backward_packet.py",
    "benchmarks/nvidia/record_e2e_spine_attention.py",
    "benchmarks/nvidia/record_e2e_spine_comparative.py",
    "benchmarks/nvidia/record_e2e_spine_epilogue.py",
    "benchmarks/nvidia/record_e2e_spine_paged_kv.py",
    "benchmarks/nvidia/record_e2e_spine_reduction.py",
    "benchmarks/nvidia/record_low_precision_native_resources.py",
    "benchmarks/nvidia/record_packed_storage_foundation.py",
    "benchmarks/nvidia/record_remaining_dtype_reduction.py",
    "benchmarks/nvidia/record_replay_parity.py",
    "benchmarks/nvidia/record_training_memory_foundation.py",
    "benchmarks/nvidia/record_transport_parity.py",
    "benchmarks/record_async_public_frames.py",
    "benchmarks/record_async_status_composition.py",
    "benchmarks/record_native_nonlinear_ad.py",
    "benchmarks/record_owned_source_state_gpu.py",
    "benchmarks/record_runtime_shape_frames.py",
    "benchmarks/record_source_exception_gpu.py",
    "benchmarks/record_source_state_gpu.py",
    "benchmarks/record_status_fanin.py",
    "benchmarks/rocm/benchmark_block_attnres_gfx1151.py",
    "benchmarks/rocm/benchmark_rocm_compiled_gemm_dtype.py",
    "benchmarks/rocm/benchmark_rocm_es_low_rank.py",
    "benchmarks/rocm/benchmark_rocm_flash_attn_bwd_compiled.py",
    "benchmarks/rocm/benchmark_rocm_raster.py",
    "benchmarks/rocm/benchmark_rocm_training_backward.py",
    "benchmarks/rocm/record_deltanet_backward_selectors.py",
    "benchmarks/spectral/benchmark_rocm_fft_plan_cache.py",
    "benchmarks/spectral/benchmark_tsol_composite.py",
    "benchmarks/x86/benchmark_x86_attention_backward_parallel.py",
    "benchmarks/x86/benchmark_x86_attention_lse.py",
    "benchmarks/x86/benchmark_x86_es_low_rank.py",
    "benchmarks/x86/benchmark_x86_fft_codelets.py",
    "benchmarks/x86/benchmark_x86_t1_cache_model.py",
    "benchmarks/x86/record_deltanet_backward_selectors.py",
})


def _recorders() -> list[Path]:
    return sorted(p for p in (ROOT / "benchmarks").rglob("*.py")
                  if "__pycache__" not in p.parts and p.name != "__init__.py" and len(p.stem) >= 6)


def _unnamed(recorders: list[Path]) -> set[str]:
    """Stems no other tracked file mentions as a whole token.

    One pass over the tracked files under the naming roots, tokenised on
    identifier characters, intersected with the stem set — a few seconds, where
    `git grep -o` over ~300 fixed strings cost two minutes on the doc tree. Two
    recorders can share a stem (`rocm/` and `x86/` both have
    `record_deltanet_backward_selectors.py`); a recorder is named only by a file
    outside that stem's own set, so siblings do not name each other.
    """
    import re

    paths_by_stem: dict[str, set[str]] = {}
    for p in recorders:
        paths_by_stem.setdefault(p.stem, set()).add(str(p.relative_to(ROOT)))
    tracked = subprocess.run(
        ["git", "ls-files", "--", *NAMING_ROOTS], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split("\n")
    token = re.compile(r"[A-Za-z0-9_]+")
    named: set[str] = set()
    self_name = f"tests/unit/{Path(__file__).name}"
    for rel in tracked:
        if not rel or rel == self_name:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for stem in set(token.findall(text)) & paths_by_stem.keys():
            if rel not in paths_by_stem[stem]:
                named.add(stem)
    return {path for stem, paths in paths_by_stem.items() if stem not in named for path in paths}


def test_every_recorder_is_named_or_frozen():
    unnamed = _unnamed(_recorders())
    new = sorted(unnamed - KNOWN_UNNAMED)
    assert not new, f"new recorder(s) nothing names — add a README row, a packet manifest or a test: {new}"
    healed = sorted(KNOWN_UNNAMED - unnamed)
    assert not healed, f"now named (or gone); remove from KNOWN_UNNAMED: {healed}"
