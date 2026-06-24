# 7.12.3. Structured Sparse Matrices

> RDNA4 ISA — pages 106–106

7.12.3. Structured Sparse Matrices
A-matrices may be sparse, where a packed (M x K/2) A-matrix represents a (M x K) matrix in which 2 out of
every 4 elements in the "K" dimension are zero. Sparse matrices use an additional set of index data which has
two 2-bit index values to define the expansion from packed to unpacked format. When the A-matrix is a 4:2
sparse matrix, the corresponding B-matrix must be (K x N), and loaded in column-major order.

Sparsity information is loaded into a single VGPR and each 2-bit index entry identifies which 2 out 4 elements
are non-zero. The two indices have the rule: Idx0 < Idx1. Sparse matrix index bits are arranged in the VGPR in
the same way that columns are arranged in the A-matrix: the values that can be found in the lane-0 VGPRs are
the same ones controlled by the index values in lane 0 of the index VGPR.

All vector-memory loads (with or without transpose) are the same as the 16x16 matrix load, except that two
load instructions are used and they load into the next consecutive group of VGPRs.

The K dimension is often 32, so expanding the A-matrix requires 16 index values to expand 16→32.
Wave32: each lane has 8 index values per lane (since the 16 columns data of are divided over 32 lanes).
Wave64: each lane has 4 index values per lane (since the 16 columns data of are divided over 64 lanes).
Sparse matrix index VGPRs contain two or four sets of index data. These may correspond to two or four A-
matrices. The VGPR holding the indices can hold 2 sets of indices in wave32 that are selected using OPSEL[0] of
the WMMA instruction, and 4 sets for wave64 selected by OPSEL[1:0]. Matrices with larger K values operate
similarly.

Expanding a packed sparse A-matrix for expanded 16x32:

    For row = 0..15
      For col = 0..31, step=4      // step through groups of 4 columns (K) in the expanded A matrix
           // Idx0 & idx1 indicate which 2 of 4 A-matrix entries are non-zero
           idxLane =   <same as VGPR layout lane#>
           idxFirstBit = isWave64 ? (col[2:2]*4) : (col[3:2]*4)
           idx0 =   VGPR[idxLane][idxFirstBit+1 : idxFirstBit+0]
           idx1 =   VGPR[idxLane][idxFirstBit+3 : idxFirstBit+2]
           idx0_value = Packed_A_matrix[row][col/2]
           idx1_value = Packed_A_matrix[row][col/2 + 1]
           Unpacked_A_Matrix[row][col+0] = idx0==0 ? idx0_value : 0
           Unpacked_A_Matrix[row][col+1] = idx0==1 ? idx0_value :
                                            idx1==1 ? idx1_value : 0;
           Unpacked_A_Matrix[row][col+2] = idx0==2 ? idx0_value :
                                            idx1==2 ? idx1_value : 0;
           Unpacked_A_Matrix[row][col+3] = idx1==3 ? idx1_value : 0;
