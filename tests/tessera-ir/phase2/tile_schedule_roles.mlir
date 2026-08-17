// RUN: tessera-opt --tessera-tile-dataflow-legality -split-input-file -verify-diagnostics %s
//
// SO-2 first carrier: both CDNA-style rotating roles and Hopper-style split
// roles use the same role -> barrier declaration and the same verifier.

func.func @ping_pong_roles() {
  %load = tile.role {name = "load_wave", kind = "producer", members = ["load"]} : !tile.role
  %compute = tile.role {name = "compute_wave", kind = "consumer", members = ["mma"]} : !tile.role
  %bar = tile.mbarrier.init with %load, %compute {slots = 2 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  return
}

// -----

func.func @hopper_split_roles() {
  %producer = tile.role {name = "tma_warp", kind = "producer", members = ["tma"]} : !tile.role
  %consumer = tile.role {name = "wgmma_warps", kind = "consumer", members = ["mma.0", "mma.1", "mma.2"]} : !tile.role
  %bar = tile.mbarrier.init with %producer, %consumer {slots = 2 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  return
}

// -----

// Roles are ordinary SSA and retain identity through an scf.for back-edge.
// Both the pipeline binding and the barrier relation are resolved to the
// declarations above the loop by TileDataflowLegalityPass.
func.func @loop_carried_roles() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %producer = tile.role {name = "rotating_load", kind = "producer", members = ["group.phase_producer"]} : !tile.role
  %consumer = tile.role {name = "rotating_mma", kind = "consumer", members = ["group.phase_consumer"]} : !tile.role
  scf.for %i = %c0 to %c2 step %c1 iter_args(%p = %producer, %c = %consumer) -> (!tile.role, !tile.role) {
    %ps = tile.pipeline_init %p {depth = 2 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
    %bar = tile.mbarrier.init with %p, %c {slots = 2 : i64, phase_bits = 1 : i64} : !tile.mbarrier
    scf.yield %p, %c : !tile.role, !tile.role
  }
  return
}

// -----

func.func @unresolved_role(%role: !tile.role) {
  // expected-error @+1 {{TILE_ROLE_RELATION_INVALID}}
  %ps = tile.pipeline_init %role {depth = 2 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  return
}

// -----

func.func @missing_consumer_role() {
  %producer = tile.role {name = "load", kind = "producer", members = ["load"]} : !tile.role
  // expected-error @+1 {{a role-bearing barrier requires both producer and consumer roles}}
  %bar = tile.mbarrier.init with %producer {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  return
}

// -----

func.func @duplicate_role_member() {
  // expected-error @+1 {{logical role members must be unique}}
  %producer = tile.role {name = "load", kind = "producer", members = ["load", "load"]} : !tile.role
  return
}
