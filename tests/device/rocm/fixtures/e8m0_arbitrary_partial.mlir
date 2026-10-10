// Native Tile diagnostic: arbitrary f32 register-owned accumulator partials.
!acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
module attributes {tessera.arch = "gfx1201", tessera.target = "rocm"} {
  gpu.module @e8m0_arbitrary_mod {
    gpu.func @e8m0_arbitrary(%p: memref<?xf32>, %sa: memref<?xi8>, %sb: memref<?xi8>,
                            %o: memref<?xf32>, %m: index, %n: index, %acc_bits: i32) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %row = arith.muli %by, %c16 : index
      %col = arith.muli %bx, %c16 : index
      %ordinal = arith.divui %by, %c16 : index
      %scalar = memref.load %p[%ordinal] : memref<?xf32>
      %vector = vector.broadcast %scalar : f32 to vector<8xf32>
      %partial = builtin.unrealized_conversion_cast %vector : vector<8xf32> to !acc
      %acc_scalar = arith.bitcast %acc_bits : i32 to f32
      %acc_vector = vector.broadcast %acc_scalar : f32 to vector<8xf32>
      %zero = builtin.unrealized_conversion_cast %acc_vector : vector<8xf32> to !acc
      %scaled = tile.fragment_scaled_accumulate %zero, %partial scales(%sa, %sb) at(%row, %col)
        group(%c0, %c1) bounds(%m, %n) {scale_n = 1 : i64, scale_format = "e8m0"}
        : !acc, memref<?xi8>, memref<?xi8>
      %tile = tile.fragment_unpack %scaled {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>}
        : (!acc) -> !tile.tile
      tile.store %tile, %o, %row, %col, %n
        {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>,
         tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>}
        : !tile.tile, memref<?xf32>, index, index, index
      gpu.return
    }
  }
}
