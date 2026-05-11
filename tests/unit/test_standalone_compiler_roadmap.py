from __future__ import annotations

from pathlib import Path

from tessera.compiler.primitive_coverage import (
    CONTRACT_FIELDS,
    all_primitive_coverages,
    coverage_for,
    coverage_summary,
    primitives_for_model_family,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "audit" / "execution_roadmap.md"
DASHBOARD = ROOT / "docs" / "audit" / "standalone_primitive_coverage.md"


def test_standalone_compiler_sprints_are_documented():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "Standalone compiler milestone sprints (S-series)" in text
    expected_sprints = [
        "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
        "S9", "S10", "S11", "S12", "S13", "S14", "S15",
    ]
    for sprint in expected_sprints:
        assert f"[{sprint}]" in text


def test_standalone_roadmap_keeps_external_frameworks_as_references_only():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "runtime-independent of PyTorch, JAX, and Flax" in text
    assert "reference vocabularies" in text
    assert "supported ops" in text


def test_s0_locks_data_pipeline_optimizers_and_aot_in_scope():
    """S0 must explicitly declare the contested boundaries as in-scope."""
    text = ROADMAP.read_text(encoding="utf-8")

    # Data pipeline is in scope per the user's S0 decision (2026-05-10).
    assert "data pipeline is in scope" in text
    assert "tf.data" in text and "torch.utils.data" in text and "grain" in text

    # Training-step elements must be in scope, otherwise S8 cannot run a
    # training step on CPU reference.
    assert "training step is in scope" in text

    # Custom-primitive authoring is the standalone differentiator.
    assert "Custom-primitive authoring is in scope" in text

    # AOT export and persistent compilation cache are deployment requirements.
    assert "AOT export and persistent compilation cache are in scope" in text


def test_standalone_roadmap_covers_required_compiler_surfaces():
    text = ROADMAP.read_text(encoding="utf-8")

    expected_surfaces = [
        "Native primitive contract registry",
        "Tensor algebra, indexing, and scalar math",
        "Pytrees, module state, and model containers",
        "Explicit RNG and stochastic effects",
        "Control flow and transform composition",
        "Native sharding, collectives, and distributed semantics",
        "Flax-level model primitive library",
        "Tiny standalone model conformance suite",
        "Numerics, mixed precision, and quantization",
        "Optimizer library and training-step primitives",
        "Loss / criterion library",
        "State serialization and checkpointing",
        "Custom-primitive / extension API",
        "Compilation cache and AOT export",
        "Native data pipeline",
    ]
    for surface in expected_surfaces:
        assert surface in text, f"missing roadmap surface: {surface}"


def test_standalone_roadmap_names_broad_model_families():
    text = ROADMAP.read_text(encoding="utf-8")

    expected_families = [
        "diffusion",
        "xLSTM",
        "Mamba",
        "Hyena",
        "Linformer",
        "cosFormer",
        "Griffin",
        "Megalodon",
        "JEPA",
        "Titans/Atlas",
    ]
    for family in expected_families:
        assert family in text


def test_primitive_coverage_imports_existing_ops_as_partial_entries():
    entries = all_primitive_coverages()

    matmul = entries["matmul"]
    assert matmul.existing_op
    assert matmul.status == "partial"
    assert matmul.graph_name == "tessera.matmul"
    # matmul has VJP+JVP registered and the contract-axis hardening passes
    # promoted math / shape / dtype / batching / transpose / lowering / tests
    # to `complete` via the category classifier. What's still incomplete is
    # the mesh-dependent sharding rule and the hardware backend kernel —
    # both gated behind Phase G integration.
    assert matmul.contract_status["lowering_rule"] == "complete"
    assert matmul.contract_status["vjp"] == "complete"
    assert matmul.contract_status["jvp"] == "complete"
    assert matmul.contract_status["batching_rule"] == "complete"
    assert matmul.contract_status["transpose_rule"] == "complete"
    assert matmul.contract_status["math_semantics"] == "complete"
    assert "vjp" not in matmul.missing_contracts()
    assert "jvp" not in matmul.missing_contracts()
    # The remaining gates for a fully contract-complete `matmul`.
    assert "sharding_rule" in matmul.missing_contracts(), (
        "sharding_rule should remain partial pending Phase G mesh integration"
    )
    assert "backend_kernel" in matmul.missing_contracts(), (
        "backend_kernel should remain partial pending Phase G hardware execution"
    )


def test_existing_coverage_consults_autodiff_vjp_registry():
    """Every op with a registered VJP must show vjp=complete in the registry."""
    from tessera.autodiff.vjp import _VJPS

    entries = all_primitive_coverages()
    # Sample a representative slice across the registered ops.
    for name in ("matmul", "softmax", "rmsnorm", "flash_attn", "fft", "rfft",
                 "gelu", "layer_norm", "dropout", "rope"):
        if name in entries and name in _VJPS:
            assert entries[name].contract_status["vjp"] == "complete", (
                f"{name} has a registered VJP but the registry reports it as "
                f"{entries[name].contract_status['vjp']}"
            )


def test_primitive_coverage_promotes_memory_primitives_from_planned():
    memory_write = coverage_for("memory_write")

    assert memory_write.existing_op
    assert memory_write.status == "partial"
    assert "Titans/Atlas" in memory_write.model_families
    assert memory_write.metadata["graph_ir_lowering"] == "stub_required"
    assert memory_write.metadata["backend_kernel"] == "reference_only"


def test_primitive_coverage_contract_fields_are_complete_for_every_entry():
    for entry in all_primitive_coverages().values():
        assert set(entry.contract_status) == set(CONTRACT_FIELDS)


def test_primitive_coverage_family_queries_and_summary():
    ssm_entries = primitives_for_model_family("Mamba/SSM")
    names = {entry.name for entry in ssm_entries}
    summary = coverage_summary()

    assert "scan" in names
    assert "selective_ssm" in names
    assert summary["partial"] > 0


def test_primitive_coverage_includes_s2_reductions_and_stability():
    """S2 must register reductions, tensor algebra, indexing, and scalar math."""
    entries = all_primitive_coverages()

    # Tensor algebra
    for name in ("reshape", "pad", "tile", "dynamic_slice", "dynamic_update_slice",
                 "cat", "stack", "split", "slice", "select", "permute", "broadcast"):
        assert name in entries, f"S2 tensor algebra missing: {name}"

    # Reductions
    for name in ("mean", "var", "argmax", "argmin", "cumsum", "cumprod",
                 "max", "min", "cummax", "cummin"):
        assert name in entries, f"S2 reduction missing: {name}"

    # Numerical-stability primitives
    for name in ("logsumexp", "log_softmax", "log1p", "expm1", "softplus"):
        assert name in entries, f"S2 stability primitive missing: {name}"

    # Comparisons + logical
    for name in ("eq", "lt", "gt", "logical_and", "logical_or", "logical_not"):
        assert name in entries, f"S2 comparison/logical missing: {name}"

    # Numeric helpers
    for name in ("clamp", "where", "isnan", "isfinite", "sign", "abs"):
        assert name in entries, f"S2 numeric helper missing: {name}"

    # Indexing / functional updates
    for name in ("gather", "scatter", "scatter_add", "scatter_reduce", "top_k",
                 "sort", "argsort", "take", "index_select", "index_update"):
        assert name in entries, f"S2 indexing/update missing: {name}"


def test_primitive_coverage_includes_s5_transforms_and_axis_helpers():
    entries = all_primitive_coverages()
    for name in ("scan", "associative_scan", "while_loop", "fori_loop",
                 "cond", "switch", "map", "value_and_grad", "vjp", "jvp",
                 "vmap", "pmap", "remat", "checkpoint", "axis_index",
                 "axis_size", "axis_name", "autocast"):
        assert name in entries, f"S5 transform missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s3_state_tree_surface():
    entries = all_primitive_coverages()
    for name in ("tree_flatten", "tree_unflatten", "tree_map", "tree_reduce",
                 "tree_transpose", "empty_state_tree", "module_state_tree",
                 "state_filter", "state_partition", "state_collection_spec"):
        assert name in entries, f"S3 state-tree primitive missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s4_rng_surface():
    entries = all_primitive_coverages()
    for name in ("rng_key", "rng_split", "rng_fold_in", "rng_clone",
                 "rng_uniform", "rng_normal", "rng_truncated_normal",
                 "rng_bernoulli", "rng_categorical", "rng_multinomial",
                 "rng_randint", "rng_permutation", "rng_gamma", "rng_beta",
                 "rng_dirichlet", "rng_poisson"):
        assert name in entries, f"S4 RNG primitive missing: {name}"


def test_primitive_coverage_includes_s6_collectives_library():
    entries = all_primitive_coverages()
    for name in ("psum", "pmean", "pmax", "pmin", "collective_permute", "broadcast_to_axis",
                 "shard_map", "named_sharding", "partition_spec"):
        assert name in entries, f"S6 collective missing: {name}"
        assert entries[name].existing_op
    assert entries["collective_permute"].category == "collective"
    assert entries["permute"].category == "tensor_algebra"


def test_primitive_coverage_includes_s7_attention_and_position_layers():
    entries = all_primitive_coverages()
    for name in ("multi_head_attention", "gqa_attention", "mqa_attention",
                 "mla_decode", "alibi", "ntk_rope", "conv1d", "max_pool",
                 "avg_pool", "gru_cell", "simple_rnn_cell", "linear_general",
                 "group_norm", "lora_linear"):
        assert name in entries, f"S7 layer/attention missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s7_memory_primitives():
    entries = all_primitive_coverages()
    for name in ("memory_read", "memory_write", "memory_evict"):
        assert name in entries, f"S7 memory primitive missing: {name}"
        assert entries[name].existing_op
        assert entries[name].category == "memory"


def test_primitive_coverage_includes_s9_quantization_and_numerics():
    entries = all_primitive_coverages()
    for name in ("quantize_int8", "dequantize_int8", "quantize_int4",
                 "dequantize_int4", "fake_quantize", "calibration_observer",
                 "grad_scaler_step"):
        assert name in entries, f"S9 quant/numerics primitive missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s8_tiny_model_conformance_targets():
    entries = all_primitive_coverages()
    for name in ("tiny_diffusion_conformance", "tiny_recurrent_conformance",
                 "tiny_attention_conformance", "tiny_training_step_conformance"):
        assert name in entries, f"S8 conformance target missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s10_optimizers_and_schedules():
    entries = all_primitive_coverages()
    for name in ("sgd", "adam", "adamw", "adafactor", "lion", "muon",
                 "cosine_lr", "cosine_warmup_lr", "linear_warmup_lr",
                 "cyclical_lr", "chained_schedule", "clip_grad_norm",
                 "clip_grad_value", "ema_update", "polyak_avg"):
        assert name in entries, f"S10 optimizer/schedule missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s11_loss_library():
    entries = all_primitive_coverages()
    for name in ("mse_loss", "mae_loss", "huber_loss", "kl_divergence",
                 "info_nce_loss", "ddpm_noise_pred_loss", "ctc_loss",
                 "binary_cross_entropy_loss", "focal_loss"):
        assert name in entries, f"S11 loss missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s12_serialization():
    entries = all_primitive_coverages()
    for name in ("save_state", "load_state", "save_sharded", "load_sharded",
                 "state_migration", "partial_state_load"):
        assert name in entries, f"S12 serialization primitive missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s13_custom_primitive_api():
    entries = all_primitive_coverages()
    for name in ("custom_primitive", "custom_call", "custom_vjp", "custom_jvp",
                 "custom_batching", "custom_lowering"):
        assert name in entries, f"S13 custom-op primitive missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s14_aot_and_cache():
    entries = all_primitive_coverages()
    for name in ("aot_export", "aot_load", "stablehlo_export",
                 "gguf_export", "safetensors_export", "compilation_cache"):
        assert name in entries, f"S14 AOT/cache primitive missing: {name}"
        assert entries[name].existing_op


def test_primitive_coverage_includes_s15_data_pipeline_and_tokenizers():
    """S0 puts the data pipeline in scope — registry must reflect S15 surface."""
    entries = all_primitive_coverages()

    # Dataset combinators
    for name in ("dataset_map", "dataset_filter", "dataset_batch",
                 "dataset_prefetch", "dataset_shuffle", "dataset_interleave",
                 "dataset_repeat", "dataset_zip", "sharded_dataset",
                 "iterable_dataset", "dataset_checkpoint"):
        assert name in entries, f"S15 dataset primitive missing: {name}"
        assert entries[name].existing_op

    # Tokenizers
    for name in ("tokenizer_byte", "tokenizer_bpe", "tokenizer_wordpiece",
                 "tokenizer_unigram", "tokenizer_sentencepiece_compat"):
        assert name in entries, f"S15 tokenizer missing: {name}"
        assert entries[name].existing_op
        assert entries[name].contract_status["vjp"] == "not_applicable"
        assert entries[name].contract_status["jvp"] == "not_applicable"


def test_remaining_data_pipeline_gaps_stay_planned():
    """Unimplemented data extensions still should not pretend to be supported."""
    entries = all_primitive_coverages()
    for name in ("dataset_take", "tokenizer_sentencepiece_training"):
        if name not in entries:
            continue
        entry = entries[name]
        assert entry.status == "planned"
        assert not entry.existing_op


def test_primitive_coverage_renders_markdown_dashboard():
    text = render_markdown([coverage_for("scan"), coverage_for("matmul")])

    assert "# Standalone Primitive Coverage" in text
    assert "`scan`" in text
    assert "`matmul`" in text
    assert "Missing contracts" in text


def test_primitive_coverage_rejects_duplicate_planned_entries(monkeypatch):
    from tessera.compiler import primitive_coverage as pc

    duplicate = pc._planned("scan", "control_flow", ("all",))
    monkeypatch.setattr(pc, "_PLANNED_ENTRIES", pc._PLANNED_ENTRIES + (duplicate,))

    try:
        pc.all_primitive_coverages()
    except ValueError as exc:
        assert "duplicate planned primitive coverage entry: scan" in str(exc)
    else:
        raise AssertionError("duplicate planned primitive entries must fail loudly")


def test_standalone_primitive_dashboard_documents_s1_contract():
    text = DASHBOARD.read_text(encoding="utf-8")

    assert "Standalone Primitive Coverage" in text
    assert "PyTorch" in text
    assert "JAX" in text
    assert "Flax" in text
    assert "Contract Axes" in text
    assert "Model-Family Coverage Tags" in text


def test_standalone_primitive_dashboard_contains_checked_generated_snapshot():
    text = DASHBOARD.read_text(encoding="utf-8")
    names = [
        "matmul",
        "permute",
        "collective_permute",
        "scan",
        "selective_ssm",
        "dataset_map",
        "tokenizer_bpe",
    ]
    generated = render_markdown([coverage_for(name) for name in names])
    generated_table = "\n".join(generated.splitlines()[5:])

    assert "<!-- BEGIN GENERATED PRIMITIVE COVERAGE SNAPSHOT -->" in text
    assert generated_table in text
