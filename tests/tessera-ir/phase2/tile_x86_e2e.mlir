// RUN: tessera-opt --tessera-tile-to-x86 %s | FileCheck %s

module {
  llvm.func @softmax(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %k: i64) {
    tile.softmax_kernel %x, %o, %rows, %k {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @reduce(%x: !llvm.ptr, %o: !llvm.ptr, %outer: i64,
                    %axis_extent: i64, %inner: i64) {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "f32", accum = "f32", kind = "mean", axis = 1 : i64,
      keepdims = false, schedule = "serial", nan_mode = "propagate",
      inner_is_one = true
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @matmul(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                    %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f32", b = "f32", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @attention(%q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr,
                       %o: !llvm.ptr, %b: i64, %h: i64, %sq: i64,
                       %sk: i64, %d: i64, %dv: i64) {
    tile.attention_kernel %q, %k, %v, %o, %b, %h, %h, %sq, %sk, %d, %dv {
      storage = "f32", accum = "f32", scale = 0.25 : f32,
      causal = true, bias = false, window_left = -1 : i64,
      window_right = -1 : i64, softcap = 0.0 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }

  llvm.func @attention_ext(%q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr,
                           %bias: !llvm.ptr, %o: !llvm.ptr, %b: i64,
                           %h: i64, %sq: i64, %sk: i64, %d: i64, %dv: i64) {
    tile.attention_kernel %q, %k, %v, %bias, %o, %b, %h, %h, %sq, %sk, %d, %dv {
      storage = "f32", accum = "f32", scale = 0.25 : f32,
      causal = false, bias = true, window_left = 3 : i64,
      window_right = 3 : i64, softcap = 4.0 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }

  llvm.func @unary(%x: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %x, %o, %n {
      family = "unary", kind = "abs", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @binary(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %a, %b, %o, %n {
      family = "binary", kind = "add", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @predicate(%x: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %x, %o, %n {
      family = "predicate", kind = "isfinite", storage = "f32",
      output_storage = "i8"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @compare(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %a, %b, %o, %n {
      family = "compare", kind = "ge", storage = "f32", output_storage = "i8"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @logical_not(%a: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %a, %o, %n {
      family = "logical", kind = "not", storage = "i8", output_storage = "i8"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @bitwise_popcount(%a: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %a, %o, %n {
      family = "bitwise", kind = "popcount", storage = "i32", output_storage = "i32"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @where(%c: !llvm.ptr, %a: !llvm.ptr, %b: !llvm.ptr,
                   %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %c, %a, %b, %o, %n {
      family = "where", kind = "where", storage = "f32",
      condition_storage = "i8", output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @transcendental(%x: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    tile.elementwise_kernel %x, %o, %n {
      family = "transcendental", kind = "exp", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @binary_math(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                         %n: i64) {
    tile.elementwise_kernel %a, %b, %o, %n {
      family = "binary_math", kind = "pow", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @matmul_bf16(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                         %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @matmul_vnni(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                         %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "u8", b = "i8", acc = "i32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @matmul_f64(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                        %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 8, n = 8, k = 8, a = "f64", b = "f64", acc = "f64", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f64">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @argreduce(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {
    tile.argreduce_kernel %x, %o, %rows, %cols {kind = "argmax", storage = "f32", output_storage = "i32", tie_break = "first"} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
  llvm.func @scan(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {
    tile.scan_kernel %x, %o, %rows, %cols {kind = "sum", storage = "f32", inclusive = true} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
  llvm.func @norm(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64, %eps: f32) {
    tile.norm_kernel %x, %o, %rows, %cols, %eps {kind = "rmsnorm", storage = "f32", accum = "f32", axis = -1 : i64, affine = false} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
  llvm.func @rope(%x: !llvm.ptr, %theta: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {
    tile.rope_kernel %x, %theta, %o, %rows, %cols {storage = "f32", layout = "interleaved_pairs"} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
  llvm.func @alibi(%slopes: !llvm.ptr, %o: !llvm.ptr, %h: i64, %s: i64) {
    tile.alibi_kernel %slopes, %o, %h, %s {storage = "f32", formula = "slope_times_j_minus_i"} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @breadth_spmm(%indptr: !llvm.ptr, %indices: !llvm.ptr,
                          %values: !llvm.ptr, %rhs: !llvm.ptr,
                          %m: i64, %n: i64, %out: !llvm.ptr) {
    tile.x86_abi_kernel %indptr, %indices, %values, %rhs, %m, %n, %out {
      symbol = "tessera_x86_avx512_spmm_csr_f32",
      abi = "tessera.x86.spmm.csr.f32.v1", family = "sparse",
      effects = "writeonly", returns_status = false
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr
    llvm.return
  }

  llvm.func @breadth_kv_append(%cache: !llvm.ptr, %max: i64, %width: i64,
                               %start: i64, %rows: !llvm.ptr, %count: i64) {
    tile.x86_abi_kernel %cache, %max, %width, %start, %rows, %count {
      symbol = "tessera_x86_kv_cache_append_f32",
      abi = "tessera.x86.kv.cache.append.f32.v1", family = "kv_cache",
      effects = "stateful", returns_status = true
    } : !llvm.ptr, i64, i64, i64, !llvm.ptr, i64
    llvm.return
  }
}

// CHECK-DAG: func.func private @tessera_x86_avx512_softmax_f32(!llvm.ptr, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_reduce_f32(!llvm.ptr, i64, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_gemm_f32(!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_flash_attn_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i32, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_flash_attn_ext_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i32, i64, f32, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_unary_f32(!llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_binary_f32(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_predicate_f32(!llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_compare_f32(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_logical_i8(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_bitwise_i32(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_where_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_transcendental_f32(!llvm.ptr, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_pow_f32(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_gemm_bf16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, f32)
// CHECK-DAG: func.func private @tessera_x86_avx512_vnni_gemm_u8s8_s32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_gemm_f64(!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_argreduce_f32(!llvm.ptr, i64, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_scan_f32(!llvm.ptr, i64, i64, !llvm.ptr, i32)
// CHECK-DAG: func.func private @tessera_x86_avx512_rmsnorm_f32(!llvm.ptr, i64, i64, f32, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_rope_f32(!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_alibi_f32(!llvm.ptr, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_avx512_spmm_csr_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr)
// CHECK-DAG: func.func private @tessera_x86_kv_cache_append_f32(!llvm.ptr, i64, i64, i64, !llvm.ptr, i64) -> i32
// CHECK-LABEL: llvm.func @softmax
// CHECK: call @tessera_x86_avx512_softmax_f32
// CHECK-NOT: tile.softmax_kernel
// CHECK-LABEL: llvm.func @reduce
// CHECK: arith.constant 2 : i32
// CHECK: call @tessera_x86_avx512_reduce_f32
// CHECK-NOT: tile.reduce_kernel
// CHECK-LABEL: llvm.func @matmul
// CHECK: call @tessera_x86_avx512_gemm_f32
// CHECK-NOT: tile.matmul_kernel
// CHECK-LABEL: llvm.func @attention
// CHECK: call @tessera_x86_flash_attn_f32
// CHECK-NOT: tile.attention_kernel
// CHECK-LABEL: llvm.func @attention_ext
// CHECK: call @tessera_x86_flash_attn_ext_f32
// CHECK-NOT: tile.attention_kernel
// CHECK-LABEL: llvm.func @unary
// CHECK: arith.constant 3 : i32
// CHECK: call @tessera_x86_avx512_unary_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @binary
// CHECK: arith.constant 4 : i32
// CHECK: call @tessera_x86_avx512_binary_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @predicate
// CHECK: arith.constant 2 : i32
// CHECK: call @tessera_x86_avx512_predicate_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @compare
// CHECK: arith.constant 5 : i32
// CHECK: call @tessera_x86_avx512_compare_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @logical_not
// CHECK: arith.constant 3 : i32
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: call @tessera_x86_avx512_logical_i8
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @bitwise_popcount
// CHECK: arith.constant 4 : i32
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: call @tessera_x86_avx512_bitwise_i32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @where
// CHECK: call @tessera_x86_avx512_where_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @transcendental
// CHECK: arith.constant 0 : i32
// CHECK: call @tessera_x86_avx512_transcendental_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @binary_math
// CHECK: call @tessera_x86_avx512_pow_f32
// CHECK-NOT: tile.elementwise_kernel
// CHECK-LABEL: llvm.func @matmul_bf16
// CHECK: arith.trunci
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: call @tessera_x86_avx512_gemm_bf16
// CHECK-NOT: tile.matmul_kernel
// CHECK-LABEL: llvm.func @matmul_vnni
// CHECK: arith.trunci
// CHECK: arith.constant 0 : i32
// CHECK: call @tessera_x86_avx512_vnni_gemm_u8s8_s32
// CHECK-NOT: tile.matmul_kernel
// CHECK-LABEL: llvm.func @matmul_f64
// CHECK: call @tessera_x86_avx512_gemm_f64
// CHECK-NOT: tile.matmul_kernel
// CHECK-LABEL: llvm.func @argreduce
// CHECK: call @tessera_x86_avx512_argreduce_f32
// CHECK-LABEL: llvm.func @scan
// CHECK: call @tessera_x86_avx512_scan_f32
// CHECK-LABEL: llvm.func @norm
// CHECK: call @tessera_x86_avx512_rmsnorm_f32
// CHECK-LABEL: llvm.func @rope
// CHECK: call @tessera_x86_avx512_rope_f32
// CHECK-LABEL: llvm.func @alibi
// CHECK: call @tessera_x86_avx512_alibi_f32
// CHECK-LABEL: llvm.func @breadth_spmm
// CHECK: call @tessera_x86_avx512_spmm_csr_f32
// CHECK-NOT: tile.x86_abi_kernel
// CHECK-LABEL: llvm.func @breadth_kv_append
// CHECK: call @tessera_x86_kv_cache_append_f32
// CHECK-NOT: tile.x86_abi_kernel
