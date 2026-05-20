<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.examples_audit --render -->

# Tessera Examples — Status Audit

This dashboard lists every active ``examples/`` entry point and its **executable status**.  It is regenerated from ``python/tessera/compiler/examples_manifest.py``.

CI guards (run as part of ``scripts/validate.sh``):

* ``python -m tessera.cli.examples_audit --check`` — executes every ``runnable`` row and ``runnable_optional`` rows whose extras are available; ``scaffold`` / ``archived`` rows are not executed.
* ``python -m tessera.cli.claim_lint --check`` — scans each example README and flags overclaim language on ``scaffold`` / ``broken`` rows.

``examples/archive/**`` is out of scope and not tracked here.

## Status taxonomy

| Status              | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| ``runnable``          | Runs on default venv + CPU-only CI.                            |
| ``runnable_optional`` | Runs when declared ``extras_required`` are importable.         |
| ``compile_only``      | Emits IR/artifacts but does not execute the workload.        |
| ``scaffold``          | Intentionally illustrative; not runnable today.              |
| ``broken``            | Expected to run, currently fails — followup needed.          |
| ``archived``          | Intentionally retired; in-tree for reference only.           |

## Counts

| Status | Count |
|--------|------:|
| ``runnable`` | 11 |
| ``runnable_optional`` | 1 |
| ``compile_only`` | 0 |
| ``scaffold`` | 6 |
| ``broken`` | 0 |
| ``archived`` | 0 |
| **total** | **18** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``examples/getting_started`` | ``runnable`` | ``examples/getting_started/basic_tensor_ops.py`` | ``PYTHONPATH=python python examples/getting_started/basic_tensor_ops.py`` |
| ``examples/getting_started/tessera_flash_attention_demo`` | ``runnable_optional`` | ``examples/getting_started/tessera_flash_attention_demo/examples/flash_attention_demo.py`` | ``PYTHONPATH=python:examples/getting_started/tessera_flash_attention_demo python examples/getting_started/tessera_flash_attention_demo/examples/flash_attention_demo.py``<br/>extras: ``torch`` |
| ``examples/compiler/ir_pipeline_tutorial`` | ``runnable`` | ``examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py`` | ``PYTHONPATH=python python examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py`` |
| ``examples/compiler/dnas`` | ``runnable`` | ``examples/compiler/dnas/dnas_schedule_autotune.py`` | ``PYTHONPATH=python python examples/compiler/dnas/dnas_schedule_autotune.py`` |
| ``examples/conformance`` | ``runnable`` | ``examples/conformance/apple_path_ga_ebm_demos.py`` | ``python examples/conformance/apple_path_ga_ebm_demos.py`` |
| ``examples/advanced/kv_cache_serving`` | ``runnable`` | ``examples/advanced/kv_cache_serving/demo.py`` | ``PYTHONPATH=python:examples/advanced/kv_cache_serving python examples/advanced/kv_cache_serving/demo.py`` |
| ``examples/advanced/long_context_attention`` | ``runnable`` | ``examples/advanced/long_context_attention/demo.py`` | ``PYTHONPATH=python:examples/advanced/long_context_attention python examples/advanced/long_context_attention/demo.py`` |
| ``examples/advanced/speculative_decoding`` | ``runnable`` | ``examples/advanced/speculative_decoding/demo.py`` | ``PYTHONPATH=python:examples/advanced/speculative_decoding python examples/advanced/speculative_decoding/demo.py`` |
| ``examples/advanced/rlvr_reasoning_suite`` | ``runnable`` | ``examples/advanced/rlvr_reasoning_suite/run_demo.py`` | ``PYTHONPATH=python:examples/advanced/rlvr_reasoning_suite python examples/advanced/rlvr_reasoning_suite/run_demo.py --steps 1 --group-size 2 --log examples/advanced/rlvr_reasoning_suite/runs/rewards.jsonl`` |
| ``examples/advanced/mla`` | ``runnable`` | ``examples/advanced/mla/tests/smoke_random.py`` | ``python examples/advanced/mla/tests/smoke_random.py`` |
| ``examples/advanced/Fast_dLLM_v2`` | ``runnable`` | ``examples/advanced/Fast_dLLM_v2/tests/smoke_random.py`` | ``python examples/advanced/Fast_dLLM_v2/tests/smoke_random.py`` |
| ``examples/advanced/Nemotron_Nano_12B_v2`` | ``runnable`` | ``examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py`` | ``python examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py`` |
| ``examples/advanced/Diffusion_LLM`` | ``scaffold`` | ``examples/advanced/Diffusion_LLM/tessera_diffusion_llm.py`` | Research sketch — references non-existent APIs (``ts.compile(mode='training')``, ``ts.randint``, ``Tensor[]`` syntax) and the package modules require PyTorch.  Reimplement against the canonical Tessera surface or mark broken when that work starts. |
| ``examples/advanced/Jet_nemotron`` | ``scaffold`` | ``examples/advanced/Jet_nemotron/examples/e2e_infer.py`` | Requires the upstream ``tessera.stdlib`` research stack which is not part of the standalone compiler surface.  Test ``tests/test_sanity.py`` locks the e2e_infer import block + skips honestly when stdlib is absent. |
| ``examples/advanced/power_retention`` | ``scaffold`` | ``examples/advanced/power_retention/examples/minimal_power_attn.py`` | Placeholder — entry-point script currently just prints ``'example'``.  Real implementation lives in the ``python/tessera_power/`` subpackage (CUDA scaffolds, Retention op) which is not wired into the audit yet. |
| ``examples/advanced/Tessera_Empirical_Software_Agent`` | ``scaffold`` | ``examples/advanced/Tessera_Empirical_Software_Agent/src/agents/tree_search_runner.py`` | End-to-end LLM + tree-search agent — requires a real LLM client, sandbox executor, and per-task harness.  DummyLLM only proposes ``print('hello from variant N')`` stubs; the orchestrator is not runnable as a CI smoke test. |
| ``examples/integration/HF_transformer`` | ``scaffold`` | ``examples/integration/HF_transformer/tessera_huggingface_transformers.py`` | References non-existent Tessera APIs (``from tessera import function, Module``); needs a rewrite against the canonical surface (``@tessera.jit`` + ``tessera.nn.Module``). |
| ``examples/optimization`` | ``scaffold`` | ``examples/optimization/README.md`` | Top-level placeholder directory with only README.md and src/ stubs — no entry-point script exists yet. |

