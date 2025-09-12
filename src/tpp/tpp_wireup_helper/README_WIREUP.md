
# TPP Wire-up (Quick Guide)

1) Unzip `tpp_package_v0_1.zip` into `src/tpp/`.
2) Apply the CMake additions from `CMakeLists_additions.txt`:
   - `add_subdirectory(src/tpp)`
   - Link `TesseraTPPDialect`, `TesseraTPPPasses`, `TesseraTPPInit` to `tessera-opt`.
3) If your driver is custom, use `register_tpp_in_driver.cpp` as a reference:
   - Create/modify your main to call `registerTPPPipelineAlias(nullptr);`
   - Make sure the TPP libs are linked so the symbol is available.
4) Build & run smoke tests:
   - `ninja tessera-opt`
   - `tessera-opt src/tpp/test/TPP/pipeline_alias.mlir -tpp-space-time`
5) Add the tests to lit by ensuring the test dir is discovered or by adding
   notes from `lit_notes.txt` to your tree-level config.

## Common Flags
- `-tpp-space-time` end-to-end pipeline alias
- `-tpp-halo-infer` halo analysis (placeholder now)
- `-lower-tpp-to-target-ir` bridge into your Target-IR

## Next
- Replace stubs with ODS-generated types/ops.
- Implement halo inference & BC lowering first for immediate value.
