// examples/mor_token_choice/mlir/mor_token_choice.mlir
//===----------------------------------------------------------------------===//
// Reference Target-IR skeleton after MoR lowering (for tests).
//===----------------------------------------------------------------------===//
t.module {
  t.global @recursion_weights : !t.params<attn,mlp> attributes { shared = true }

  t.func @recursion_block(%x: tensor<?x?x?xf32>, %kv: !t.kvcache,
                          %w: !t.params<attn,mlp>,
                          %cfg: !t.cfg { heads = 4, causal = true })
      -> tensor<?x?x?xf32> {
    %y = t.attention %x, %kv, %w { heads = 4, causal = true, tile = [64,64,64] }
         : tensor<?x?x?xf32> -> tensor<?x?x?xf32>
    %z = t.mlp %y, %w { activation = "gelu" }
         : tensor<?x?x?xf32> -> tensor<?x?x?xf32>
    %o = arith.addf %z, %x : tensor<?x?x?xf32>
    return %o : tensor<?x?x?xf32>
  }

  t.func @mor_main() -> i32 {
    %S = t.const 3 : i32
    %h0 = t.randn {seed = 123} : tensor<2x64x256xf32>
    %depth = t.mor.route_token_choice %h0 { max_depth = 3 } : tensor<2x64xi32>
    %kv_all = t.kv.alloc { policy = "recursion", heads = 4, d_model = 256, seq_len = 64 }

    %h = %h0
    %s = t.const 1 : i32
    scf.while (%s) : (i32) -> (i32) {
      %cond = arith.cmpi sle, %s, %S : i32
      scf.condition(%cond) %s : i32
    } do {
      %active = t.mor.partition %h, %depth, %s
               : (tensor<2x64x256xf32>, tensor<2x64xi32>, i32) -> tensor<?x?x?xf32>
      %kv_s = t.kv.view %kv_all, %s : (!t.kvcache.all, i32) -> !t.kvcache
      %w = t.address_of @recursion_weights : !t.params<attn,mlp>
      %y = call @recursion_block(%active, %kv_s, %w, %cfg)
           : (tensor<?x?x?xf32>, !t.kvcache, !t.params<attn,mlp>, !t.cfg)
             -> tensor<?x?x?xf32>
      %h = t.mor.scatter %h, %y, %depth, %s
           : (tensor<2x64x256xf32>, tensor<?x?x?xf32>, tensor<2x64xi32>, i32)
             -> tensor<2x64x256xf32>
      %s_next = arith.addi %s, (t.const 1 : i32) : i32
      scf.yield %s_next : i32
    }

    t.debug.emit_tag "MOR_DEPTH_COUNTS", %depth : tensor<2x64xi32>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
