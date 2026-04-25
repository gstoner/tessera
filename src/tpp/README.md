
# Tessera TPP (v0.2)
What's new:
- **ODS types/attrs**: !tpp.field, !tpp.mesh, #tpp.units, #tpp.bc (definitions + printers/parsers via tablegen).
- **Halo inference pass**: `-tpp-halo-infer` annotates ops with `tpp.halo`.
- **BC lowering sketch**: marks masked-store lowering for `tpp.bc.enforce` in `-lower-tpp-to-target-ir`.
- **Examples & CI**: shallow-water smoke test; GitHub Actions workflow running `-tpp-space-time`.

> Note: ODS-generated headers require the usual MLIR tablegen steps (CMake targets provided).
