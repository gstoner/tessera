module { "tessera_rocm.flash_attn"() {name = "attention_forward", head_dim = 64 : i64, dtype = "f16", arch = "gfx1201", save_lse = true, half_fragment_loads = true} : () -> () }
