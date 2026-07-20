// RUN: tessera-opt %s --split-input-file --verify-diagnostics -o /dev/null

module {
  llvm.func @bad_storage(%x: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    // expected-error @+1 {{'tile.elementwise_kernel' op unary requires storage="f32"}}
    tile.elementwise_kernel %x, %o, %n {
      family = "unary", kind = "abs", storage = "f16",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// -----

module {
  llvm.func @bad_logical_storage(%a: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    // expected-error @+1 {{'tile.elementwise_kernel' op logical requires storage="i8"}}
    tile.elementwise_kernel %a, %o, %n {
      family = "logical", kind = "not", storage = "f32",
      output_storage = "i8"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// -----

module {
  llvm.func @bad_bitwise_output(%a: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    // expected-error @+1 {{'tile.elementwise_kernel' op bitwise requires output_storage="i32"}}
    tile.elementwise_kernel %a, %o, %n {
      family = "bitwise", kind = "popcount", storage = "i32",
      output_storage = "i8"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// -----

module {
  llvm.func @bad_predicate_output(%x: !llvm.ptr, %o: !llvm.ptr, %n: i64) {
    // expected-error @+1 {{'tile.elementwise_kernel' op predicate requires output_storage="i8"}}
    tile.elementwise_kernel %x, %o, %n {
      family = "predicate", kind = "isfinite", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// -----

module {
  llvm.func @bad_binary_kind(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                            %n: i64) {
    // expected-error @+1 {{'tile.elementwise_kernel' op binary kind must be sub|div|maximum|minimum|add|mul|mod|floor_div}}
    tile.elementwise_kernel %a, %b, %o, %n {
      family = "binary", kind = "pow", storage = "f32",
      output_storage = "f32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}
