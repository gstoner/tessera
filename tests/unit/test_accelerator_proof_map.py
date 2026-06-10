"""Contract + drift tests for the S-series accelerator-proof map.

The map classifies every primitive-coverage entry into exactly one accelerator
class (proven / eligible / special / multi_device / host) and renders
docs/audit/generated/s_series_accelerator_proof.md. These tests lock the
invariants and the proven⇔envelope correspondence so the map can't silently
drift from the runtime envelope or the coverage registry.
"""

from __future__ import annotations

from pathlib import Path

from tessera.compiler import accelerator_proof as ap
from tessera.compiler import driver, op_catalog, primitive_coverage


def test_every_primitive_classified_exactly_once():
    rows = ap.all_rows()
    names = [r.name for r in rows]
    assert len(names) == len(set(names))  # no dups
    assert {r.name for r in rows} == set(primitive_coverage.all_primitive_coverages())
    valid = set(ap._CLASS_ORDER)
    for r in rows:
        assert r.accel_class in valid, (r.name, r.accel_class)


def test_summary_partitions_the_registry():
    s = ap.summary()
    assert sum(s.values()) == len(primitive_coverage.all_primitive_coverages())
    assert set(s) == set(ap._CLASS_ORDER)


def test_proven_iff_graph_name_in_apple_gpu_envelope():
    env = driver._APPLE_GPU_RUNTIME_OPS
    for r in ap.all_rows():
        gn = op_catalog.graph_name_for(r.name)
        in_env = gn is not None and gn in env
        assert (r.accel_class == "proven") == in_env, (
            f"{r.name}: class={r.accel_class} graph_name={gn} in_env={in_env}")


def test_proven_count_matches_live_envelope():
    proven = [r for r in ap.all_rows() if r.accel_class == "proven"]
    env = driver._APPLE_GPU_RUNTIME_OPS
    # every proven primitive's graph name resolves into the envelope
    assert all(op_catalog.graph_name_for(r.name) in env for r in proven)
    assert ap.summary()["proven"] == len(proven)


def test_collectives_are_multi_device():
    rows = {r.name: r for r in ap.all_rows()}
    for op in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
        if op in rows:
            assert rows[op].accel_class == "multi_device", op


def test_host_only_categories_have_no_flop_to_accelerate():
    # A spot-check that genuinely structural categories land in `host`.
    rows = {r.name: r for r in ap.all_rows()}
    for op in ("tree_flatten", "dataset_batch", "aot_export", "constant_lr"):
        if op in rows:
            assert rows[op].accel_class == "host", op


def test_known_proven_ops_are_proven():
    # Ops this session routed to metal_runtime must show proven.
    rows = {r.name: r for r in ap.all_rows()}
    for op in ("clifford_geometric_product", "ebm_energy_quadratic",
               "contrastive_divergence_loss", "z_loss", "selective_ssm"):
        if op in rows:
            assert rows[op].accel_class == "proven", op


def test_dashboard_in_sync():
    # The checked-in dashboard equals a fresh render (drift gate).
    here = Path(__file__).resolve().parents[2]
    path = here / "docs" / "audit" / "generated" / "s_series_accelerator_proof.md"
    rendered = ap.render_markdown()
    on_disk = path.read_text()
    assert on_disk.rstrip("\n") == rendered.rstrip("\n"), (
        "s_series_accelerator_proof.md is stale — regenerate via "
        "`python -c 'from tessera.compiler.accelerator_proof import write_dashboard; write_dashboard()'`")


def test_spectral_fft_lane_is_proven():
    # A (2026-06-10): the 9 spectral ops moved special -> proven once the
    # MPSGraph FFT lane landed and they joined the apple_gpu envelope.
    rows = {r.name: r.accel_class for r in ap.all_rows()}
    for op in ("fft", "ifft", "rfft", "irfft", "dct", "stft", "istft",
               "spectral_conv", "spectral_filter"):
        assert rows[op] == "proven", op


def test_rng_non_kernel_ops_reclassified_host():
    # B (2026-06-10): RNG key-management is host bookkeeping and the MCMC
    # samplers are host orchestration — neither is a dedicated-kernel candidate,
    # so they must NOT inflate the `special` (device-RNG kernel gap) count.
    rows = {r.name: r.accel_class for r in ap.all_rows()}
    for op in ("rng_key", "rng_split", "rng_fold_in", "rng_clone",
               "rng_gibbs_sample", "rng_hmc_sample",
               "rng_langevin_sample", "rng_mala_sample"):
        assert rows[op] == "host", op


def test_special_is_exactly_the_device_rng_sampler_kernels():
    # After A + B, `special` is precisely the genuine device-RNG sampler kernels.
    special = {r.name for r in ap.all_rows() if r.accel_class == "special"}
    expected = {
        "rng_uniform", "rng_normal", "rng_bernoulli", "rng_beta", "rng_gamma",
        "rng_poisson", "rng_dirichlet", "rng_categorical", "rng_multinomial",
        "rng_truncated_normal", "rng_randint", "rng_permutation",
    }
    assert special == expected, sorted(special ^ expected)
