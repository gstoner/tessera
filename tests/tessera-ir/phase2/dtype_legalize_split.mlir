// RUN: tessera-opt --allow-unregistered-dialect --tessera-compute-legalize %s | FileCheck %s --check-prefix=COMPUTE
// RUN: tessera-opt --allow-unregistered-dialect --tessera-compute-legalize --tessera-storage-legalize %s | FileCheck %s --check-prefix=STORAGE
// RUN: tessera-opt --allow-unregistered-dialect --tessera-compute-legalize --tessera-storage-legalize --tessera-ir-contracts %s | FileCheck %s --check-prefix=CONTRACTS
//
// C4 (2026-06-23, TIRx review / COMPILER_AUDIT item C4): Decision #15a as pass
// ordering. compute-legalize (early) gives reduced-precision storage a wide
// accumulator; storage-legalize (terminal) packs sub-byte storage. The third
// RUN proves the legalized IR then passes IRContractLegalityPass (the static
// verifier) — assign + verify, the two-sided contract.

// bf16 storage, no accumulator → compute-legalize stamps accum = fp32.
// COMPUTE-LABEL: func.func @bf16_gemm
// COMPUTE: numeric_policy = {accum = "fp32", storage = "bf16"}
// CONTRACTS-LABEL: func.func @bf16_gemm
func.func @bf16_gemm(%a: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "bf16"}}
       : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %c : tensor<4x4xbf16>
}

// int8 storage → the wide accumulator is int32, not fp32.
// COMPUTE-LABEL: func.func @int8_gemm
// COMPUTE: numeric_policy = {accum = "int32", storage = "int8"}
func.func @int8_gemm(%a: tensor<4x4xi8>) -> tensor<4x4xi8> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "int8"}}
       : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi8>
  return %c : tensor<4x4xi8>
}

// fp4 storage → compute-legalize adds accum = fp32; storage-legalize then packs
// it into an int8 container (terminal step).
// COMPUTE-LABEL: func.func @fp4_gemm
// COMPUTE: numeric_policy = {accum = "fp32", storage = "fp4_e2m1"}
// STORAGE-LABEL: func.func @fp4_gemm
// STORAGE: tessera.storage_container = "int8"
// STORAGE: tessera.storage_packed = true
// CONTRACTS-LABEL: func.func @fp4_gemm
func.func @fp4_gemm(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp4_e2m1"}}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// fp32 storage is not reduced-precision → untouched (no accum stamped).
// COMPUTE-LABEL: func.func @fp32_passthrough
// COMPUTE: numeric_policy = {storage = "fp32"}
func.func @fp32_passthrough(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp32"}}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// Already-legalized policy → idempotent (accum unchanged, not duplicated).
// COMPUTE-LABEL: func.func @already_has_accum
// COMPUTE: numeric_policy = {accum = "fp32", storage = "bf16"}
func.func @already_has_accum(%a: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {accum = "fp32", storage = "bf16"}}
       : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %c : tensor<4x4xbf16>
}
