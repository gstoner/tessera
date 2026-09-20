// RUN: tessera-opt --split-input-file --verify-diagnostics %s | FileCheck %s
//
// `tile.fragment_pack {transpose}` states that the source tile arrives in the
// opposite major order and the lowering owes the transpose. It exists so a
// b-role fragment can be packed from a ROW-major [K, N] staging buffer, which
// is what lets the global read, the LDS write and the fragment read all be
// contiguous at once (docs/backends/rocm/wmma-fragment-layout.md 10j.1, 10s).
//
// Decision #21a: it selects semantics, so it fails closed -- a backend that
// cannot transpose must reject rather than ignore it, since ignoring it
// silently computes a different matrix.

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
!fb = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "wmma">
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// A b-role pack MAY carry it.
// CHECK-LABEL: @b_role_accepts_transpose
// CHECK: tile.fragment_pack
// CHECK-SAME: transpose
func.func @b_role_accepts_transpose(%t: !tile.tile) -> !fb {
  %f = tile.fragment_pack %t {transpose, mma = #mma, tile.layout = #layout} : (!tile.tile) -> !fb
  return %f : !fb
}

// -----

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
!fa = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "wmma">
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// An a-role pack MUST NOT. `a` already wants the major order a row-major
// buffer supplies, so a transpose here is a request nobody has a reason to
// honour -- and a backend would have to guess what was meant.
func.func @a_role_rejects_transpose(%t: !tile.tile) -> !fa {
  // expected-error @+1 {{transpose is only meaningful for a b-role fragment}}
  %f = tile.fragment_pack %t {transpose, mma = #mma, tile.layout = #layout} : (!tile.tile) -> !fa
  return %f : !fa
}
