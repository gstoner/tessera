// RUN: tessera-opt %s -split-input-file -verify-diagnostics

// MLIR 23 exposes signless i64 attributes through unsigned value accessors.
// These cases pin the verifier contract to the signed IntegerAttr spelling so
// negative values cannot wrap into apparently valid large positive values.

func.func @shared_positive_i64_rejects_negative(
    %x: tensor<4x8xf32>, %w: tensor<8x1xf32>) -> tensor<4x1xf32> {
  // expected-error @+1 {{max_depth must be positive}}
  %d = "tessera.mor_router"(%x, %w) {max_depth = -1 : i64}
      : (tensor<4x8xf32>, tensor<8x1xf32>) -> tensor<4x1xf32>
  return %d : tensor<4x1xf32>
}

// -----

func.func private @while_body()
func.func private @while_cond()
func.func @control_while_rejects_negative(%init: tensor<1x4xf32>)
    -> tensor<1x4xf32> {
  // expected-error @+1 {{max_iters must be positive}}
  %r = "tessera.control_while"(%init) {
      body = @while_body, cond = @while_cond, carry_arg_index = 0 : i64,
      max_iters = -1 : i64
    } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----

func.func @mor_partition_rejects_negative(
    %x: tensor<4x8xf32>, %depth: tensor<4xi64>) -> tensor<4xi1> {
  // expected-error @+1 {{step must be non-negative}}
  %mask = "tessera.mor_partition"(%x, %depth) {step = -1 : i64}
      : (tensor<4x8xf32>, tensor<4xi64>) -> tensor<4xi1>
  return %mask : tensor<4xi1>
}

// -----

func.func @write_row_rejects_negative(
    %buffer: tensor<8x4xf32>, %value: tensor<4xf32>) -> tensor<8x4xf32> {
  // expected-error @+1 {{row must be non-negative}}
  %result = "tessera.write_row"(%buffer, %value) {row = -1 : i64}
      : (tensor<8x4xf32>, tensor<4xf32>) -> tensor<8x4xf32>
  return %result : tensor<8x4xf32>
}

// -----

func.func @kv_prune_rejects_negative(%cache: !tessera.kv_cache)
    -> !tessera.kv_cache {
  // expected-error @+1 {{window must be positive}}
  %updated = "tessera.kv_cache.prune"(%cache) {window = -1 : i64}
      : (!tessera.kv_cache) -> !tessera.kv_cache
  return %updated : !tessera.kv_cache
}

// -----

func.func @flash_attn_rejects_negative(
    %q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>,
    %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  // expected-error @+1 {{head_dim must be positive}}
  %o = "tessera.flash_attn"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}>
      {head_dim = -1 : i64}
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}

// -----

func.func @hybrid_attention_rejects_negative(
    %q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>,
    %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  // expected-error @+1 {{layer_index must be non-negative}}
  %o = "tessera.hybrid_attention"(%q, %k, %v) {layer_index = -1 : i64}
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}

// -----

func.func @attn_boundary_rejects_negative(
    %scores: tensor<8x16xf32>, %q: index, %kv: index)
    -> tensor<8x16xf32> {
  // expected-error @+1 {{logical_sk must be positive}}
  %masked = "tessera_attn.boundary_mask"(%scores, %q, %kv) {
      causal = true, window_left = -1 : i64, window_right = -1 : i64,
      logical_sk = -1 : i64
    } : (tensor<8x16xf32>, index, index) -> tensor<8x16xf32>
  return %masked : tensor<8x16xf32>
}

// -----

func.func @attn_causal_rejects_negative(%scores: tensor<8x16xf32>)
    -> tensor<8x16xf32> {
  // expected-error @+1 {{q_offset and kv_offset must be non-negative}}
  %masked = "tessera_attn.causal_mask"(%scores) {
      q_offset = -1 : i64, kv_offset = 0 : i64
    } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %masked : tensor<8x16xf32>
}
