# Checked positive-stride ROCm paged-KV execution

Owner: E2E-REAL-6. Synchronization: ROCM-STRIDED-PAGED-KV-20261008.

This packet proves static rank-4 f32 pages with four positive whole-element
strides through public `@jit`, serialized native packages, prepared/resident
execution and native HIP graph replay. The page table is compact i32; gathered
output is compact f32. Padded/offset, permuted and Fortran views execute without
compacting the borrowed source. Read-only self-aliasing views are supported
when their complete addressed span fits a proved backing allocation.

## Compiler and runtime boundary

Python/textual frontend -> typed Graph MLIR -> native Schedule MLIR -> Tile
MLIR -> ROCm Target IR -> LLVM/HSACO -> checked native runtime ABI -> HIP.
The distinct ABI is `tessera.rocm.paged_kv.pages_table_o_dims_strides.f32_i32.v1`:
three buffers, seven extent scalars and four source element strides. Native
passes generate the physical gather addresses; Python supplies storage facts.
Schedule identity seals positive-stride and checked-span policies. Runtime
pitch values do not enter image identity. Prepared owners seal their actual
pitch and source span; resident storage and image leases remain native-owned.

Capacity, overflow, forged views, pitch mismatch, output writability and output
aliasing are checked before GPU reads/writes. Negative/zero/fractional strides,
other page dtypes, symbolic extents and arbitrary device-buffer carriers remain
outside this proved envelope. This is one named route, not closure of the
entire paged-KV family, E2E-REAL-6 or the five-slice program.

## Exact-device validation

- Princess-Luna: gfx1151, AMD Radeon 8060S, PCI `0000:c5:00.0`.
  Final owning gate: **52 passed, 4 other-architecture cases skipped**.
- Tajasaurus: gfx1201, AMD Radeon RX 9070 XT, PCI `0000:03:00.0`.
  Final owning gate: **48 passed, 6 other-architecture/family cases skipped**.
  Its pytest environment reports an unregistered timeout configuration warning.
- Super-Bear WSL: 140 focused host tests passed, 32 hardware cases skipped in
  the earlier integration sweep; the final alias test was added afterward and
  executed on both owning devices. Mypy reported zero errors in all six touched
  package/runtime files. Host checks are not ROCm device proof.

Owning tests cover bit-exact independent NumPy gather oracles, public and
serialized replay, warm compiler-forbidden reuse, retained results, sealed
stride mismatch, read-only sources, invalid output borrowing and native graph
capture. The self-alias test has a 64-byte backing span for 1,536 logical bytes.
A native paged-KV -> softmax chain checks the compact private intermediate
against an independent float64 oracle and captures two actual kernel nodes.
The existing MoE compact-layout refusal remains tested.

Core and ROCm tools were built on Super-Bear against assertion-enabled
LLVM/MLIR 23.1.1, then installed unchanged on both GPU hosts with the matching
layout library. The HIP movement runtime was freshly built on each owning
host; the pre-existing image-cache runtime ABI was retained. This is not a
claim of a fresh full compiler rebuild on each GPU host. Packets contain the
actual compiler, target compiler and runtime SHA-256 values and verify all
17 recorded source hashes against the delivered product/recorder bytes.

## Correctness-gated timings

Recorder: `benchmarks/rocm/benchmark_strided_paged_kv.py`. Each host records
seven rotating rounds with 32 repetitions. All eight cases pass numerical
checks before and after timing and have zero warm compiler subprocess calls.
Each architecture reuses one strided image across three layouts and two shapes;
compact pages have their separate existing image. Saved stage MLIR (lossless deterministic gzip) and raw
samples accompany each packet.

Small: pages `(4,256,3,8)`, 4 logical pages, 1,024 output tokens.
Large: pages `(32,16,8,128)`, 64 logical pages, 1,024 output tokens.
The four layouts within a profile use identical logical page values.

Kernel time uses HIP events around one resident kernel and excludes transfers.
Public time measures completed JIT calls including upload/download and
synchronization, excluding compilation. These scopes must not be compared as
a speedup ratio. Completed timing windows are at least 52 ms on gfx1151 and
62 ms on gfx1201. No selector/default-route promotion is made.

| Architecture | Profile | Layout | Kernel median (us) | Public median (ms) |
| --- | --- | --- | ---: | ---: |
| gfx1151 | small | compact | 11.641 | 1.9431 |
| gfx1151 | small | padded | 11.561 | 2.0248 |
| gfx1151 | small | permuted | 11.481 | 2.0197 |
| gfx1151 | small | fortran | 11.441 | 2.0051 |
| gfx1151 | large | compact | 32.021 | 2.4750 |
| gfx1151 | large | padded | 42.623 | 2.8114 |
| gfx1151 | large | permuted | 86.966 | 2.6195 |
| gfx1151 | large | fortran | 86.965 | 2.6958 |
| gfx1201 | small | compact | 8.720 | 2.2629 |
| gfx1201 | small | padded | 11.280 | 2.5763 |
| gfx1201 | small | permuted | 11.360 | 2.5342 |
| gfx1201 | small | fortran | 11.400 | 2.5190 |
| gfx1201 | large | compact | 26.800 | 2.6661 |
| gfx1201 | large | padded | 33.520 | 2.5712 |
| gfx1201 | large | permuted | 56.240 | 2.5874 |
| gfx1201 | large | fortran | 59.379 | 2.5553 |

The public call remains roughly 2–3 ms, much larger than device execution.
The larger permuted/Fortran layouts also cost more kernel time than compact
storage. Host-launch/tracing attribution and physical coalescing optimization
remain open; these measurements establish correctness and cost, not closure
of ROCm performance obligations or sibling-backend execution.

## Receipts and reproduction

`gfx1151/` and `gfx1201/` each retain `benchmark-delivery/device.json`, saved
Graph/Schedule/Tile/Target/backend stage files (`.mlir.gz`), final owning-test output and
native runtime build output. Use the matching owning toolchain/runtime and set
`TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1`, `TESSERA_ROCM_CHIP` to the live
architecture, and compiler/runtime paths before running:

```sh
python -m pytest -q tests/unit/test_strided_paged_kv_runtime.py \
  tests/unit/test_public_movement_frontend.py \
  tests/unit/test_native_graph_rocm_movement.py
python benchmarks/rocm/benchmark_strided_paged_kv.py --architecture gfx1151 \
  --output "$HOME/scratch/strided-paged-kv-proof" --rounds 7 --repeats 32
```

The final compiler/span/backing/ABI gate passed 47 host tests with ten hardware
skips; all saved stage bytes match the corresponding lineage SHA-256 after
gzip decompression. Final audit/diagnostic/pass metadata gates passed 307 host tests. The focused
host, mypy and final drift logs are retained under `host/`.

All validation ran through authorized host WSL/ROCm environments. Full-unit
generic scaled-matmul batching/transpose closure remains a separate red gate;
these focused tests do not replace it.
