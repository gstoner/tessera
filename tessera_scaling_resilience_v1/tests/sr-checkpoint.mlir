// RUN: tessera-opt-sr -tessera-insert-recompute %s | FileCheck %s
tessera_sr.checkpoint {
  // CHECK: tessera_sr.checkpoint
}