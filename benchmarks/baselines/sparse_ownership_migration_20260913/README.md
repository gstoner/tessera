# Logical sparse, ownership and floor migration — 2026-09-13

Bounded correctness evidence on Tajasarus: Radeon RX 9070 XT gfx1201,
ROCm 10.0, LLVM 23.1.1 with assertions, and the owning Zen 5 x86 CPU.
This packet records uncommitted work after PR #747; source/image/compiler hashes
bind individual numerical results. It contains no selector-grade timing.

- `sparse/*.json`: six f16/bf16 matrix cases, M/N/K = 16/16/32,
  32/48/64 and 48/32/128. Compiled Schedule → Tile → Target → HSACO performs
  logical A/B packing, sparse-index selection and K-loop accumulation.
  Every numerical output matches the f32 oracle exactly; disassembly contains
  the expected SWMMAC instruction. Invalid groups in the final K tile produce
  failure status for each affected output-column tile.
- `x86_floor.json`: three native descriptor executions through the registered
  floor Graph op and replay-bound Schedule/Tile contract, using the freshly built
  shipped x86 elementwise runtime. Signed zeros, subnormals, fractions and
  infinities match bitwise; NaN classification matches. No NaN payload claim.
- `reader_device_tests.txt`: four gfx1201 external-copy/retirement checks
  across saved/recompute policies and synchronous/asynchronous reader release.
  The owner rejects device-wide synchronization.
- `device_tests.txt`: sparse device tests, native replay tests and host reader
  interleavings. The host ownership tests are not device-overlap evidence.
- `route_census.json`: lexical caller candidates and surviving local emitter
  paths. Graph annotations include migrated dispatch branches; these counts
  are neither remaining-family counts nor physical certificates.

Reproduce sparse proof in owning WSL with `TESSERA_GFX1201_DEVICE_PROOF=1`
using `tests/unit/test_rocm_sparse_logical.py`. Optionally set
`TESSERA_SPARSE_PROOF_DIR` to retain the image/source digest records. Reproduce
floor with `benchmarks/record_absolute_migration.py --operation floor --output
<packet.json>` and the built `TESSERA_X86_ELEMENTWISE_LIB`.

Open: public sparse Graph capture and validity-consuming runtime binding,
additional formats, general attention AD ownership, isolated resident-frame
teardown recovery, and calibrated runtime/kernel attribution. The sparse producer
supports f32 accumulation only. It does not silently prune dense inputs. No
hardware performance promotion or evidence transfer to another architecture.
