# Knowledge Index

Generated from docs/context/ontology.yaml, docs/context/knowledge_map.yaml, and tests/COMPILER_TEST_PLAN.md. This is a derived navigation artifact; canonical specs remain authoritative.

Use this generated index for navigation, then verify claims against the canonical docs.

## api

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `api.python_frontend` | Python Frontend API | normative | docs/CANONICAL_API.md, docs/spec/PYTHON_API_SPEC.md, python/tessera/__init__.py | defined_by -> doc.canonical_api; implemented_in -> python/tessera/compiler/jit.py; tested_by -> test.unit; example.getting_started_basic_tensor_ops -> depends_on; test.numerical_validation -> depends_on |

## backend

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `backend.nvidia_gpu` | NVIDIA GPU Backend | normative | docs/spec/TARGET_IR_SPEC.md, python/tessera/compiler/gpu_target.py, src/runtime/src/backend/cuda_backend.cpp | tested_by -> test.mlir_lit; pass.lower_to_gpu -> depends_on |
| `backend.rocm` | ROCm Backend | normative | docs/spec/COMPILER_REFERENCE.md, src/compiler/codegen/Tessera_ROCM_Backend/CMakeLists.txt, src/runtime/src/backend/hip_backend.cpp | none |
| `backend.tpu` | TPU StableHLO Backend | normative | docs/spec/COMPILER_REFERENCE.md, python/tessera/compiler/tpu_target.py, tests/tessera-ir/phase4/tpu_shardy_export.mlir | none |
| `backend.x86` | x86 AMX/AVX512 Backend | normative | docs/spec/COMPILER_REFERENCE.md, src/compiler/codegen/tessera_x86_backend/CMakeLists.txt, src/runtime/src/backend/cpu_backend.cpp | tested_by -> test.unit |

## concept

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `concept.diagnostics` | Diagnostics And Error Handling | normative | docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md, python/tessera/diagnostics.py, tests/unit/test_error_reporter.py | explained_by -> docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md |
| `concept.shape_layout` | Shape And Layout System | normative | docs/spec/SHAPE_SYSTEM.md, docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md, python/tessera/shape.py | defined_by -> docs/spec/SHAPE_SYSTEM.md |

## doc

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `doc.documentation_authority_tree` | Documentation Authority Tree | normative | docs/README.md | doc.compiler_reference -> phase_status |
| `doc.canonical_api` | Canonical API | normative | docs/CANONICAL_API.md | api.python_frontend -> defined_by; supersedes -> docs/archive/pre_canonical/api |
| `doc.compiler_reference` | Compiler Reference | normative | docs/spec/COMPILER_REFERENCE.md | phase_status -> doc.documentation_authority_tree |
| `doc.glossary` | Glossary | informative | docs/GLOSSARY.md | none |
| `doc.lowering_pipeline_spec` | Lowering Pipeline Spec | normative | docs/spec/LOWERING_PIPELINE_SPEC.md | none |
| `doc.python_api_spec` | Python API Spec | normative | docs/spec/PYTHON_API_SPEC.md | none |

## example

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `example.getting_started_basic_tensor_ops` | Basic Tensor Ops Example | informative | examples/getting_started/basic_tensor_ops.py | depends_on -> api.python_frontend |
| `example.ir_pipeline_tutorial` | IR Pipeline Tutorial | informative | examples/compiler/ir_pipeline_tutorial/README.md, examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py | depends_on -> ir.graph |

## ir_layer

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `ir.graph` | Graph IR | normative | docs/spec/GRAPH_IR_SPEC.md, src/compiler/ir/TesseraOps.td, python/tessera/compiler/graph_ir.py | example.ir_pipeline_tutorial -> depends_on; defined_by -> docs/spec/GRAPH_IR_SPEC.md; lowers_to -> ir.schedule |
| `ir.schedule` | Schedule IR | normative | docs/spec/TARGET_IR_SPEC.md, docs/architecture/Compiler/Tessera_Compiler_ScheduleIR_Design.md, src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td | ir.graph -> lowers_to; lowers_to -> ir.tile |
| `ir.tile` | Tile IR | normative | docs/spec/TILE_IR.md, docs/spec/TARGET_IR_SPEC.md, src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td | ir.schedule -> lowers_to; lowers_to -> ir.target |
| `ir.target` | Target IR | normative | docs/spec/TARGET_IR_SPEC.md, docs/architecture/Compiler/Tessera_Compiler_TargetIR_Design.md, src/compiler/mlir/lib/Target/TesseraTargetIR.cpp | ir.tile -> lowers_to; pass.lower_to_x86 -> depends_on |

## pass

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `pass.lower_to_gpu` | tessera-lower-to-gpu | normative | docs/spec/LOWERING_PIPELINE_SPEC.md, docs/spec/COMPILER_REFERENCE.md, tests/tessera-ir/phase3/tile_ir_lowering.mlir | depends_on -> backend.nvidia_gpu |
| `pass.lower_to_x86` | tessera-lower-to-x86 | normative | docs/spec/LOWERING_PIPELINE_SPEC.md, docs/spec/COMPILER_REFERENCE.md, tests/tessera-ir/phase2/full_pipeline.mlir | depends_on -> ir.target |

## runtime_component

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `runtime.c_abi` | Runtime C ABI | normative | docs/spec/RUNTIME_ABI_SPEC.md, src/runtime/include/tessera/tessera_runtime.h, src/runtime/src/tessera_runtime.cpp | defined_by -> docs/spec/RUNTIME_ABI_SPEC.md |
| `runtime.profiler` | Tessera Profiler | informative | docs/guides/Tessera_Profiling_And_Autotuning_Guide.md, tools/profiler/README.md, python/tessera/profiler.py | explained_by -> docs/guides/Tessera_Profiling_And_Autotuning_Guide.md |

## test_suite

| ID | Name | Authority | Primary References | Relations |
| --- | --- | --- | --- | --- |
| `test.mlir_lit` | MLIR Lit Tests | normative | tests/tessera-ir, tests/tessera-ir/lit.cfg.py | backend.nvidia_gpu -> tested_by |
| `test.numerical_validation` | Numerical Validation | normative | tests/tessera_numerical_validation, tests/tessera_numerical_validation/README.md | depends_on -> api.python_frontend |
| `test.performance` | Performance Tests | normative | tests/performance, benchmarks/run_all.py | none |
| `test.unit` | Unit Tests | normative | tests/unit, tests/COMPILER_TEST_PLAN.md | api.python_frontend -> tested_by; backend.x86 -> tested_by |
