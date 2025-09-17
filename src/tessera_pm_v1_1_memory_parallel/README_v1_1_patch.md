# Tessera v1.1 Patch — Memory/Execution + Parallelism
Generated: 20250917_213209

Contents:
- `docs/Memory_Execution_Model_v1_1.md`, `docs/Parallelism_Constructs_v1_1.md`
- IR ODS stubs: Tile memory ops, cache ops, mesh/pipeline ops, MoE hooks
- lit tests in `tests/pm_v1_1/`
- Pipeline alias stub `tools/tessera-opt/PassPipelinesPM11.cpp`
- Minimal `CMakeLists.txt`

How to use:
1. Drop these into your repo, then wire dialect registration and real verifiers.
2. Hook `-pm-v1_1-verify` / `-pm-v1_1-legalize` to your tessera-opt.
3. Run lit tests to validate the surfaces.
