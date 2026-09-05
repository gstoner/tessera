// RUN: %tessera_strict_opt --tessera-tile-barrier-reuse-legality -split-input-file -verify-diagnostics %s

func.func @carried_generation_completes(%n: index) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %first = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%previous = %first) -> (!tile.async_token) {
    tile.wait_async %previous : (!tile.async_token) -> ()
    %next = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
    scf.yield %next : !tile.async_token
  }
  tile.wait_async %last : (!tile.async_token) -> ()
  tile.dealloc %a : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----
func.func @unforwarded_generation_stays_pending(%cond: i1) {
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
  %local = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
    // expected-note @+1 {{previous write to the same allocation root}}
    %t = tile.async_copy %local, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
    scf.if %cond {
      tile.wait_async %t : (!tile.async_token) -> ()
    }
  }
  // expected-error @+1 {{is reused before completion}}
  tile.buffer_write %src {tile.layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  return
}

// -----
func.func @branch_result_completion(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %result = scf.if %cond -> (!tile.async_token) {
    %t = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
    scf.yield %t : !tile.async_token
  } else {
    %t = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
    scf.yield %t : !tile.async_token
  }
  tile.wait_async %result : (!tile.async_token) -> ()
  tile.dealloc %a : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----
func.func @cfg_forwards_allocation_and_completion() {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %t = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  cf.br ^next(%a, %t : !tile.buffer, !tile.async_token)
^next(%alias: !tile.buffer, %done: !tile.async_token):
  tile.wait_async %done : (!tile.async_token) -> ()
  tile.dealloc %alias : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----
func.func @cfg_loop_generations(%cond: i1) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %first = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  cf.br ^loop(%first : !tile.async_token)
^loop(%previous: !tile.async_token):
  tile.wait_async %previous : (!tile.async_token) -> ()
  %next = tile.async_copy %a, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
  cf.cond_br %cond, ^exit(%next : !tile.async_token), ^loop(%next : !tile.async_token)
^exit(%last: !tile.async_token):
  tile.wait_async %last : (!tile.async_token) -> ()
  tile.dealloc %a : !tile.buffer
  tile.dealloc %src : !tile.buffer
  return
}

// -----
func.func @opaque_origin_in_branch_is_rejected(%cond: i1, %opaque: !tile.async_token) {
  %a = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %t = tile.async_copy %a : (!tile.buffer) -> !tile.async_token
  %selected = scf.if %cond -> (!tile.async_token) {
    scf.yield %t : !tile.async_token
  } else {
    scf.yield %opaque : !tile.async_token
  }
  // expected-error @+1 {{!tile.async_token operand must be produced by a tile.async_copy}}
  tile.wait_async %selected : (!tile.async_token) -> ()
  return
}

// -----
// Waiting on the last result cannot retire an earlier, overwritten generation
// from the SAME static copy operation. Each iteration has a fresh destination,
// but all generations borrow %src until their own completion.
func.func @last_generation_does_not_release_overwritten_copy() {
  %src = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
  %initial = tile.async_copy %src : (!tile.buffer) -> !tile.async_token
  tile.wait_async %initial : (!tile.async_token) -> ()
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %last = scf.for %i = %c0 to %c2 step %c1 iter_args(%unused = %initial) -> (!tile.async_token) {
    %local = tile.alloc {bytes = 64 : i64, space = "smem", layout = #tile.layout<shard = [8] : [1] on ["m"], replica = [] : [] on [], offset = 0>} : !tile.buffer
    // expected-note @+1 {{previous write to the same allocation root}}
    %next = tile.async_copy %local, %src : (!tile.buffer, !tile.buffer) -> !tile.async_token
    scf.yield %next : !tile.async_token
  }
  tile.wait_async %last : (!tile.async_token) -> ()
  // expected-error @+1 {{is deallocated before completion}}
  tile.dealloc %src : !tile.buffer
  return
}
