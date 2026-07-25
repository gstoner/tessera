// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=PACK
//
// C4 reconciliation (2026-06-23): the backend-neutral storage-pack descriptor
// `tessera.storage_pack = #tile.packed_format<...>` (emitted by the
// StoragePackConsume pass) now drives the ROCm WMMA kernel generator — one
// packing contract feeds both AMD (here) and NVIDIA. The descriptor's `logical`
// selects the dtype, and its `factor` IS the WMMA integer pack mode (int4 -> 2
// nibble-pack, int8 -> 1), so the two must agree.

// The descriptor (logical=int4, factor=2) drives the int4 kernel — the iu4 ABI
// fragment is vector<2xi32> (16 nibble-packed int4 per lane).
// PACK: gpu.module @g_mod
// PACK: gpu.func @g
// PACK: vector<2xi32>
"tessera_rocm.wmma_gemm"() {name = "g", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 16 : i64, nt = 16 : i64,
  tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "signed_twos_complement", encoding = "twos_complement", lane_order = "low_to_high">} : () -> ()

// -----

// A descriptor whose factor (1) disagrees with the int4 WMMA pack mode (2) is
// caught — the dtype contract and the WMMA ABI have drifted.
// expected-error @+1 {{TILE_PACKED_FORMAT_INVALID}}
"tessera_rocm.wmma_gemm"() {name = "g", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 16 : i64, nt = 16 : i64,
  tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 1, signedness = "signed_twos_complement", encoding = "twos_complement", lane_order = "scalar_lsb">} : () -> ()

// -----

// Unsigned IU4 is a distinct unregistered policy and cannot alias signed int4.
// expected-error @+1 {{TILE_PACKED_FORMAT_INVALID}}
"tessera_rocm.wmma_gemm"() {name = "g", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 16 : i64, nt = 16 : i64,
  tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "unsigned", encoding = "twos_complement", lane_order = "low_to_high">} : () -> ()
