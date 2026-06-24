# 7.12. Wave Matrix Multiply Accumulate (WMMA)

> RDNA4 ISA — pages 99–99

Instruction                 Index                  Function
V_MOVRELD_B32               M0[31:0]               Move with relative destination:
                                                   VGPR[dst + M0[31:0]] = VGPR[src]
V_MOVRELS_B32                                      Move with relative source:
                                                   VGPR[dst] = VGPR[src + M0[31:0]]
V_MOVRELSD_B32                                     Move with relative source and destination:
                                                   VGPR[dst + M0[31:0]] = VGPR[src + M0[31:0]]
V_MOVRELSD_2_B32            Src: M0[9:0]           Move with relative source and destination, each different:
                            Dst: M0[25:16]         VGPR[dst + M0[25:16]] = VGPR[src + M0[9:0]]
V_SWAPREL_B32                                      Swap two VGPRs, each relative to a separate index:
                                                   tmp = VGPR[src + M0[9:0]]
                                                   VGPR[src + M0[9:0]] = VGPR[dst + M0[25:16]]
                                                   VGPR[dst + M0[25:16]] = tmp

7.12. Wave Matrix Multiply Accumulate (WMMA)
Wave Matrix Multiply-Accumulate (WMMA) instructions provide acceleration for matrix-multiplication
operations. Each WMMA or SWMMAC (Sparse WMMA) instruction performs a single matrix multiply
operation with the data in VGPRs holding one each of the A, B, C and D-matrices. One matrix is striped across all
of the lanes - it’s not one matrix per lane. The instructions are encoded using the VOP3P encoding.

These perform: A * B + C ⇒ D, where A, B, C and D are matrices.

Simplified example of matrix multiplication on 4x4 matrices:

   Additional information can be found on the GPUOpen blog:
   https://gpuopen.com/learn/wmma_on_rdna3/
   This blog post pertains to RDNA3 but may aid in understanding RDNA4.

   The AMD Matrix Instruction Calculator:
   (https://github.com/ROCm/amd_matrix_instruction_calculator)
   contains a helper tool that allows developers to view detailed information about the matrix instructions
   in the RDNA4 architecture. It allows users to query instruction-level information such as computational
   throughput and register usage. It also allows users to generate mappings between matrix element and
   hardware registers for each matrix instruction and their modifiers.

SWMMAC instructions take a sparse matrix for matrix A (matrix B must be dense). Sparse matrices consume
half the storage space by requiring that two out of every 4 elements be zero. Index sets are included to specify
