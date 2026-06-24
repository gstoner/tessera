# 7.12.2. Matrix Element Storage in VGPRs

> RDNA4 ISA — pages 101–105

First Instruction   Second Instruction                                   Requirement between First and Second Inst
WMMA                WMMA where matrix A, B or index are the same         At least 1 V_NOP or independent VALU
                    as or overlap with previous WMMA instruction’s       instruction. This is required for correct function.
                    D-matrix.
WMMA                WMMA instruction with same VGPR of previous          Stall if the first and second instruction are not the
                    WMMA instruction’s Matrix D as Matrix C              same type of WMMA or use ABS/NEG on SRC2 of
                                                                         the second instruction.
WMMA                WMMA instruction with overlapped VGPR of             Hardware may stall.
                    previous WMMA instruction’s Matrix D as Matrix
                    C
WMMA                VALU instruction reads the previous WMMA             Hardware may stall.
                    instruction’s Matrix D
WMMA                WMMA instruction reads the VALU’s result as          Hardware may stall.
                    Matrix A/B/C

7.12.2. Matrix Element Storage in VGPRs
This section describes in detail where each element in a matrix is stored: which lane and which VGPR. The
tables below shows how matrix elements are stored in VGPRs for various data-sizes and wave-sizes. "Verilog"-
style notation is used to represent bit extractions.

Simplified view of matrix layout in VGPRs:
The A matrix is laid out in VGPRs such that one row of data is striped across VGPRs within one lane (although
due to storage limitations it may wrap into another lane). The B, C and D matrices are laid out in the opposite
manner: one row of the matrix is striped across the lanes within one VGPR.

The next section provides the details of this layout.

All layouts follow the below procedure to map a matrix element referenced by its row and column to a specific
VGPR address, lane index and bit offset within the VGPR. For sparse matrices this procedure is applied on the
pre-expanded matrix data.

   row=0..31 col=0..15

   Memory[row][col] → VGPR[lane][vgpr][startPosn*dataSize + dataSize-1 : startPosn*dataSize ]

       StartPosn is scaled by dataSize to indicate where within a VGPR DWORD the data is placed.
       E.g. dataSize=8 and startPosn=2 means data is in bits: [23:16].

            Data Size Field              Wave32                              Wave64
                                                      A-Matrix 16x16 (M x K) Matrix
            16         lane              { col[2], row[3:0] }                { col[3:2], row[3:0] }
                       vgpr              { col[3], col[1] }                  col[1]
                       startPosn         col[0]                              col[0]
            8          lane              { col[3], row[3:0] }                { col[2], col[3], row[3:0] }
                       vgpr              col[2]                              0
                       startPosn         col[1:0]                            col[1:0]
            4          lane              { col[3], row[3:0] }                {0, col[3], row[3:0]}
                       vgpr              0                                   0
                       startPosn         col[2:0]                            col[2:0]

            Data Size Field              Wave32                              Wave64
                                                     A-Matrix 16x32 (M x K) Matrix
            4          lane              { col[4], row[3:0] }                {col[3], col[4], row[3:0]}
                       vgpr              col[3]                              0
                       startPosn         col[2:0]                            col[2:0]
                                                     B-Matrix 16x16 (K x N) Matrix
            16         lane              { row[2], col[3:0] }                { row[3:2], col[3:0] }
                       vgpr              { row[3], row[1] }                  row[1]
                       startPosn         row[0]                              row[0]
            8          lane              { row[3], col[3:0] }                { row[2], row[3], col[3:0] }
                       vgpr              row[2]                              0
                       startPosn         row[1:0]                            row[1:0]
            4          lane              { row[3], col[3:0] }                { 0, row[3], col[3:0] }
                       vgpr              0                                   0
                       startPosn         row[2:0]                            row[2:0]
                                                     B-Matrix 32x16 (K x N) Matrix
            4          lane              { row[4], col[3:0] }                { row[3], row[4], col[3:0] }
                       vgpr              row[3]                              0
                       startPosn         row[2:0]                            row[2:0]
                                                          Matrix C and D (16x16)
            32         lane              { row[3], col[3:0] }                { row[2], row[3], col[3:0] }
                       vgpr              row[2:0]                            row[1:0]
                       startPosn         0                                   0
            16         lane              { row[3], col[3:0] }                { row[2], row[3], col[3:0] }
                       vgpr              row[2:1]                            row[1]
                       startPosn         row[0]                              row[0]

    • WMMA_16X16X16_IU4 : works with wave64 but uses only lanes 0..31; others unused.
    • WMMA_16X16X32_IU4 : works with wave32 but uses 2 VGPRs

                            Table 42. ISA Operand Fields
Instruction Type     Operand           Meaning
WMMA / SWMMAC        SRC0              A-Matrix
WMMA / SWMMAC        SRC1              B-Matrix
WMMA                 SRC2              C-Matrix
SWMMAC               SRC2              Sparse Index Data
WMMA                 VDST              D-Matrix
SWMMAC               VDST              C-Matrix and D-Matrix.
                                       SWMMAC reads VDST and accumulates into it.

                                         Table 43. Matrix Sizes and VGPR Usage
Data Size Wave      A-Matrix Size packed A-Matrix Size          Dense A,B VGPRs                 Sparse # VGPRs
          size      (MxK)                Expanded
16        32        16x16                    16x32              A=B=4                           A=4, B=8, S=0.5
          64        16x16                    16x32              A=B=2                           A=2, B=4, S=0.25
8         32        16x16                    16x32              A=B=2                           A=2, B=4, S=0.5
          64        16x16                    16x32              A=B=1                           A=1, B=2, S=0.25

Data Size Wave      A-Matrix Size packed A-Matrix Size     Dense A,B VGPRs               Sparse # VGPRs
          size      (MxK)                Expanded
4         32        16x16                 16x32            A=B=1 from first subv         A=1, B=2, S=0.5
                                                           only
          64        16x16                 16x32            A=B=1 from first subv         A=0.5, B=2, S=0.25
                                                           only
4         32        16x32                 16x64            A=B=2                         A=2, B=4, S=1
          64        16x32                 16x64            A=B=1                         A=1, B=2, S=0.5

    "subv" = sub-vector (32 lanes of a wave64)
      ◦ "first subv" = lanes 0..31
    "S" = sparse matrix index VGPRs
    "0.5" = just lanes 0..31 of the wave64’s lanes 0..63, or for wave32 uses just lanes 0..15
    "0.25" = uses just 16 lanes of 64.

Example: 16x16 A-Matrix of 16-bit data in Row-Major order:

This diagram shows the layout of an A-matrix in memory. The number shown in each element is the WORD-
address (i.e. byte address divided by 2).

Layout of that same matrix in VGPRs:

Register format for dense matrix results:

In order to store the matrices C and D (the result of a WMMA/SWMMAC operation), the rows of the matrix are
split into chunkSize = waveSize / 16 chunks. In other words, the rows are split in half for wave32 mode and
into four equal parts for wave64 mode. The lanes of the wave grouped into chunks of 16 lanes, with each lane
holding chunkSize = 16 / (waveSize / 16) values.

For wave32 mode, the first chunk (lanes 0..15) receives the first half of each row (indices 0..7) and the second
chunk (lanes 16..31) receives the second half (indices (8..15). In wave64 mode, the chunks are permuted so that
the first 32 lanes are the even chunks and the second 32 lanes have the odd chucks. That is, lanes 0..15 receive
m = 0..3, lanes 16..31 have m=8..11, lanes 32..47 have m=4..7, and lanes 48..63 have m=12..15.

The columns of the result matrix are spread across the 16 lanes in each chunk. Each lane holds chunkSize
elements from its row, packed contiguously into VGPRs. In the case of 16-bit matrix outputs, such as for the
WMMA_F16_16x16x16_F16 instruction, two result elements are packed into each result VGPR, with even elements
in the lower 16 bits.

That is, for 32-bit matrix results, this is the map:

  matrix[row][col] -> VGPR[(row / 8) * 16 + col][row % 8]

and for 16-bit outputs, the map is:

  matrix[row][col] -> VGPR[(row / 8) * 16 + col][(row % 8) / 2][15 + (row % 2) * 16:(row % 2) * 16]

In wave64 mode, the following is the map for 32-bit outputs:

    matrix[row][col] -> VGPR[(row / 4) % 2 * 32 + (row / 8) * 16 + col][row % 4]

and for 16-bit outputs, the map is:

    matrix[row][col] -> VGPR[(row / 4) % 2 * 32 + (row / 8) * 16 + col][(row % 4) / 2][15 + (row % 2) *
    16:(row % 2) * 16]

For example, the result layout of WMMA_F32_16x16x16xF16 in wave32 mode, showing "D[M, N]":

   VGPR:           V0                  V1             …                 V7
LANE
0                  D[0, 0]             D[1, 0]        …                 D[7, 0]
1                  D[0, 1]             D[0, 1]        …                 D[7, 1]
…                  …                   …              …                 …
15                 D[0, 15]            D[1, 15]       …                 D[7, 15]
16                 D[8, 0]             D[9, 0]        …                 D[15, 0]
…                  …                   …              …                 …
31                 D[8, 15]            D[9, 15]       …                 D[15, 15]

and the layout for WMMA_F16_16x16x16_F16 for wave32 mode is:

   VGPR:           V0[15:0]            V0[31:16]      …                 V3[31:16]
LANE
0                  D[0, 0]             D[1, 0]        …                 D[7, 0]
1                  D[0, 1]             D[1, 1]        …                 D[7, 1]
…                  …                   …              …                 …
15                 D[0, 15]            D[1, 15]       …                 D[7, 15]
16                 D[8, 0]             D[9, 0]        …                 D[15, 0]
…                  …                   …              …                 …
31                 D[8, 15]            D[9, 15]       …                 D[15, 15]

In wave64 mode, the layout for WMMA_F32_16x16x16_F16 changes to:

   VGPR:           V0                  V1             V2                V3
LANE
0                  D[0, 0]             D[1, 0]        D[2, 0]           D[3, 0]
1                  D[0, 1]             D[1, 1]        …                 D[3, 1]
…                  …                   …              …                 …
15                 D[0, 15]            D[1, 15]       D[2, 15]          D[3, 15]
16                 D[8, 0]             D[9, 0]        D[10, 0]          D[11, 0]
…                  …                   …              …                 …
32                 D[4, 0]             D[5, 0]        D[6, 0]           D[7, 0]
…                  …                   …              …                 …
63                 D[12, 15]           D[13, 15]      D[14, 15]         D[15, 15]
