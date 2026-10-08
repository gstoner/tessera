# gfx1201 mapped product/sum native integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-COMPOSED-SCALED-MAPS-2026-10-08.

72 owning RX 9070 XT cases pass ordinary compiler-owned primal, scale JVP and reverse AD. Three map policies (all, scales-only, LHS-only), one/two leading axes, independent/shared scales and K256/K37 are covered. Independent float64 per-plane oracles accumulate unmapped adjoints across mapped planes. Changed inputs/seeds/cotangents and retained output lifetimes pass; warm compiler/reference calls are forbidden.

The initial ordinary primal failure exposed a target-family executor handoff bug. The corrected artifact selects rocm while retaining gfx1201 architecture/evidence metadata. Both initial and terminal logs are retained. Device pytest warns that its timeout plugin is unavailable; numerical assertions execute, timeout enforcement is not claimed.

## Separate timings

| Mode | Policy/depth | MNK | Native event median ms | Public host median ms |
| --- | --- | --- | ---: | ---: |
| primal | all/1 | [17, 19, 256] | 0.020639 | 1.669564 |
| primal | scales/2 | [3, 5, 256] | 0.048782 | 1.807817 |
| primal | lhs/2 | [3, 5, 37] | 0.054585 | 1.765193 |
| forward | all/1 | [17, 19, 256] | 0.063882 | 2.870792 |
| forward | scales/2 | [3, 5, 256] | 0.076499 | 2.982505 |
| forward | lhs/2 | [3, 5, 37] | 0.090366 | 3.027489 |
| reverse | all/1 | [17, 19, 256] | 8.523332 | 10.836383 |
| reverse | scales/2 | [3, 5, 256] | 0.608519 | 2.684023 |
| reverse | lhs/2 | [3, 5, 37] | 1.110239 | 3.429898 |

Five windows of 100 prepared whole-program invocations measure HIP events; five separate windows of eleven warm public calls include host copies and completion. Every profile passes numerics before and after timing. No comparison or speedup is claimed; serial adjoint remains default.

## Identity and scope

Device: AMD Radeon RX 9070 XT; live architecture: gfx1201; HIP UUID hex: 32386439653765666266326566373136.
Compiler SHA256: 6075bf1450770b22635f9a22959638e4845287030d716d37c513cd7a8db51eb0.
Packet binds recorder, oracle, frontend, runtime, images and program source snapshots. Evidence annotations added to capability/execution registries after recording do not alter the tested native images.

General nonleading/output-axis maps, product-output broadcasting, dynamic/completely arbitrary composition, storage derivatives and generic batching/transpose closure remain open. This is gfx1201 evidence only. Apple/x86 require owning parity; NVIDIA needs its own scaled-contract implementation and owning proof. No full-unit green or universal compiler closure is claimed.

Host registry pre-regeneration record: 376 passed, one dashboard drift failure. Generated dashboard repaired through owning generator; final gates recorded separately.

Final host WSL focused gates: **554 passed** in 29.70 seconds, including mapped contracts, runtime/capability/diagnostic/pass registries, canonical dtype, operation foundation, tensor attributes and audit documents. This does not replace a full unit run.
