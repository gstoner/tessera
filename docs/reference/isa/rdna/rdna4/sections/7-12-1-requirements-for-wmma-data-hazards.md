# 7.12.1. Requirements for WMMA data hazards

> RDNA4 ISA — pages 100–100

which 2 elements out of every 4 are zero.

                                            Table 41. WMMA Instructions
           Instruction                                Matrix A      Matrix B       Matrix C    Result Matrix
           V_WMMA_F32_16X16X16_F16                    16x16 F16     16x16 F16      16x16 F32   16x16 F32
           V_WMMA_F32_16X16X16_BF16                   16x16 BF16 16x16 BF16 16x16 F32          16x16 F32
           V_WMMA_F16_16X16X16_F16                    16x16 F16     16x16 F16      16x16 F16   16x16 F16
           V_WMMA_BF16_16X16X16_BF16                  16x16 BF16 16x16 BF16 16x16 BF16 16x16 BF16
           V_WMMA_I32_16X16X16_IU8                    16x16 IU8     16x16 IU8      16x16 I32   16x16 I32
           V_WMMA_I32_16X16X16_IU4                    16x16 IU4     16x16 IU4      16x16 I32   16x16 I32
           V_WMMA_I32_16X16X32_IU4                    16x32 IU4     32x16 IU4      16x16 I32   16x16 I32
           V_WMMA_F32_16X16X16_FP8_FP8                16x16 FP8     16x16 FP8      16x16 F32   16x16 F32
           V_WMMA_F32_16X16X16_FP8_BF8                16x16 FP8     16x16 BF8      16x16 F32   16x16 F32
           V_WMMA_F32_16X16X16_BF8_FP8                16x16 BF8     16x16 FP8      16x16 F32   16x16 F32
           V_WMMA_F32_16X16X16_BF8_BF8                16x16 BF8     16x16 BF8      16x16 F32   16x16 F32
                                               Sparse Matrix Operations
                                   (A-Matrix size shown is after sparse data expansion)
           V_SWMMAC_F32_16X16X32_F16                  16x32 F16     32x16 F16      16x16 F32   16x16 F32
           V_SWMMAC_F32_16X16X32_BF16                 16x32 BF16 32x16 BF16 16x16 F32          16x16 F32
           V_SWMMAC_F16_16X16X32_F16                  16x32 F16     32x16 F16      16x16 F16   16x16 F16
           V_SWMMAC_BF16_16X16X32_BF16                16x32 BF16 32x16 BF16 16x16 BF16 16x16 BF16
           V_SWMMAC_I32_16X16X32_IU8                  16x32 IU8     32x16 IU8      16x16 I32   16x16 I32
           V_SWMMAC_I32_16X16X32_IU4                  16x32 IU4     32x16 IU4      16x16 I32   16x16 I32
           V_SWMMAC_I32_16X16X64_IU4                  16x64 IU4     64x16 IU4      16x16 I32   16x16 I32
           V_SWMMAC_F32_16X16X32_FP8_FP8              16x32 FP8     32x16 FP8      16x16 F32   16x16 F32
           V_SWMMAC_F32_16X16X32_FP8_BF8              16x32 FP8     32x16 BF8      16x16 F32   16x16 F32
           V_SWMMAC_F32_16X16X32_BF8_FP8              16x32 BF8     32x16 FP8      16x16 F32   16x16 F32
           V_SWMMAC_F32_16X16X32_BF8_BF8              16x32 BF8     32x16 BF8      16x16 F32   16x16 F32

"IU4" and "IU8" mean that the operand is either signed or unsigned (4 or 8 bits) as indicated by the NEG bits
instead of performing negation.

The NEG[1:0] field is repurposed for the "IU" integer types to indicate whether the inputs are signed or not
(0=unsigned, 1=signed). For WMMA using IU8/IU4 NEG[0] indicates if the A-matrix is signed, NEG[1] indicates
if the B-matrix is signed; NEG[2] and NEG_HI[2:0] must be zero. The destination is signed for the integer types.

For WMMA using F16/BF16, NEG[1:0] is applied on SRC1 and SRC0’s low 16bit. NEG_HI[1:0] is applied on SRC1
and SRC0’s high 16bit. {NEG_HI[2], NEG[2]} is applied on SRC2, and acts as {ABS, NEG}.

For WMMA using FP8/BF8, NEG must be set to zero for the A- and B-matrices.

7.12.1. Requirements for WMMA data hazards
In the table below "WMMA" is either WMMA or SWMMAC, and "V_NOP" means either a V_NOP instruction or
another unrelated (independent) VALU instruction in between the first and second instruction in the table.

First Instruction   Second Instruction                                  Requirement between First and Second Inst
                                  The cases below are required for correct function
