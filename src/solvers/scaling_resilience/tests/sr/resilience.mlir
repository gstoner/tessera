// RUN: tessera-opt -tessera-resilience-restart %s | FileCheck %s --check-prefix=RST
%t = tessera_sr.resilience_region {
  // body
} : !tessera_sr.token
// RST: sr.restart_token