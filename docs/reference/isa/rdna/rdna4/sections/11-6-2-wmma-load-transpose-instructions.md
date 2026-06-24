# 11.6.2. WMMA Load-Transpose Instructions

> RDNA4 ISA — pages 152–153

normally, and those that are out-of-range are ignored.

Out-of-range VSRC VGPRs for stores:
each DWORD of the instruction is individually checked for out-of-range. Those that are in-range supply data as
specified, and those out of range read data from VGPR0.

11.6. WMMA Matrix Load Ops with Transpose
Matrices are often stored in memory in a tiled manner. RDNA4 has instructions that simplify loading of 16x16
matrix tiles into VGPRs where the matrix in memory has the opposite major order (row vs. column) as required
by the WMMA operations. The instructions load and transpose a block of memory into VGPRs.

See WMMA Ops for information on WMMA matrix storage in VGPRs and matrix multiply operations.

11.6.1. Dense Matrices
An "A-matrix" of (M rows x K columns) is composed of elements (1 or 2 bytes each for this discussion) arranged
in memory in either row-major or column-major. The examples below show the layout for a 16x16 matrix.

Row Major Data in memory for A-matrix (MxK)

  Memory_address = (col# + row# * K) * ElementSize (# of bytes)

Column Major Data in memory for B-matrix (KxN)

  Memory_address = (col# * K + row#) * ElementSize (# of bytes)

11.6.2. WMMA Load-Transpose Instructions
These two instructions perform a load of an 8-bit or 16-bit 16x16 matrix and transpose row and column major
order.

If EXEC==0, the instruction acts like an S_NOP; otherwise these instructions require that EXEC be set to all
ones, else the operation is undefined.

The diagrams below show which matrix element each of the 32 lanes in a wave32 loads in a A-matrix. E.g. lane
0 loads 64 bits of contiguous memory and stores it in the matrix: K=0, M=0..7. B-matrix loads are similar: one
lane loads multiple contiguous N-values along single K-dimension index.

Instructions                Description
GLOBAL_LOAD_TR_B128         Load a 16x16 matrix of 16-bit data into VGPRs and transpose between row-major and column-
                            major order.
                            This instruction loads the same amount of data for both wave32 and wave64:
                            wave32 loads data into 4 consecutive VGPRs;
                            wave64 loads data into 2 consecutive VGPRs but use only addresses from lanes 0-31 (that
                            refers to 128-bit data) and ignore addresses in lanes 32-63.
                            This instruction behaves similarly to GLOBAL_LOAD_B128 except that after the 16
                            consecutive memory bytes are read (per lane), they are transposed before being stored into
                            VGPRS: instead of filling two consecutive VGPRs, they fill 16-bits in each of 8 VGPRs.
GLOBAL_LOAD_TR_B64          Load a 16x16 matrix of 8-bit data into VGPRs and transpose between row-major and column-
                            major order.
                            This instruction loads the same amount of data for both wave32 and wave64:
                            wave32 loads data into 2 consecutive VGPRs;
                            wave64 loads data into 1 VGPR but use only addresses from lanes 0-31 (that refers to 64-bit
                            data) and ignore addresses in lanes 32-63. This instruction behaves similarly to
                            GLOBAL_LOAD_B64 except that after the 8 consecutive memory bytes are read (per lane),
                            they are transposed before being stored into VGPRS: instead of filling two consecutive VGPRs,
                            they fill 16-bits in each of 8 VGPRs.

All fields of these instructions are identical to GLOBAL_LOAD_B64 and _B128, and as loads they are tracked
with LOADcnt.

Memory order      Element    Wave         VGPR Layout    Instruction to use
                  Size       Size
Row Major         16         32           Row Major      GLOBAL_LOAD_B128
Row Major         16         64           Row Major      GLOBAL_LOAD_B64
Column Major      16         32           Row Major      GLOBAL_LOAD_TR_B128 — writes 128 bits per lane x 32 lanes
Column Major      16         64           Row Major      GLOBAL_LOAD_TR_B128 — writes 64bits per lane x 64 lanes
Row Major         8          32           Row Major      GLOBAL_LOAD_B64
Row Major         8          64           Row Major      GLOBAL_LOAD_B32
Column Major      8          32           Row Major      GLOBAL_LOAD_TR_B64 — writes 64 bits per lane x 32 lanes
Column Major      8          64           Row Major      GLOBAL_LOAD_TR_B64 — writes 32bits per lane x 64 lanes

  • The above table can also be used when "VGPR Layout" is column-major: simply reverse the "memory
    order" meaning between "Row" and "Column".
