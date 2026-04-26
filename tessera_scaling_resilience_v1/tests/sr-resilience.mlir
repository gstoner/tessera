// RUN: tessera-opt-sr -tessera-resilience-restart %s | FileCheck %s
%t = tessera_sr.resilience_region {
} : !tessera_sr.token
// CHECK: tessera_sr.resilience_region