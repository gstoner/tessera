# 6.4. Integer Arithmetic Instructions

> RDNA3.5 ISA — pages 59–59

6.3. Scalar Condition Code (SCC)
The scalar condition code (SCC) is written as a result of executing most SALU instructions. For integer
arithmetic it is used as carry/borrow in for extended integer arithmetic.

The SCC is set by many instructions:
  • Compare operations: 1 = true.
  • Arithmetic operations: 1 = carry out.
     ◦ SCC = overflow for signed add and subtract operations. For add ops, overflow = both operands are of
        the same sign, and the MSB (sign bit) of the result is different than the sign of the operands. For
        subtract (A - B), overflow = A and B have opposite signs and the resulting sign is not the same as the
        sign of A.
  • Bit/logical operations: 1 = result was not zero.

6.4. Integer Arithmetic Instructions
This section describes the arithmetic operations supplied by the SALU. The table below shows the scalar
integer arithmetic instructions:

                                         Table 21. Integer Arithmetic Instructions
Instruction               Encoding        Sets SCC?       Operation
S_ADD_I32                 SOP2            Ovfl            D = S0 + S1, SCC = overflow.
S_ADD_U32                 SOP2            Cout            D = S0 + S1, SCC = carry out.
S_ADDC_U32                SOP2            Cout            D = S0 + S1 + SCC, SCC = overflow.
S_SUB_I32                 SOP2            Ovfl            D = S0 - S1, SCC = overflow.
S_SUB_U32                 SOP2            Cout            D = S0 - S1, SCC = carry out.
S_SUBB_U32                SOP2            Cout            D = S0 - S1 - SCC, SCC = carry out.
S_ADD_LSH{1,2,3,4}_U32 SOP2               D!=0            D = S0 + (S1 << {1,2,3,4})
S_ABSDIFF_I32             SOP2            D!=0            D = abs (S0 - S1), SCC = result not zero.
S_MIN_I32                 SOP2            D!=0            D = (S0 < S1) ? S0 : S1
S_MIN_U32                                                 SCC = (S0 < S1)
S_MAX_I32                 SOP2            D!=0            D = (S0 > S1) ? S0 : S1
S_MAX_U32                                                 SCC = (S0 > S1)
S_MUL_I32                 SOP2            No              D = S0 * S1 low 32bits of result
                                                          works identically for unsigned data
S_ADDK_I32                SOPK            Ovfl            D = D + simm16, SCC = overflow. Sign extended version of
                                                          simm16.
S_MULK_I32                SOPK            No              D = D * simm16. Return low 32bits. Sign extended version of
                                                          simm16.
S_ABS_I32                 SOP1            D!=0            D.i = abs (S0.i). SCC=result not zero.
S_SEXT_I32_I8             SOP1            No              D = { 24{S0[7]}, S0[7:0] }.
S_SEXT_I32_I16            SOP1            No              D = { 16{S0[15]}, S0[15:0] }.
S_MUL_HI_I32              SOP2            No              D = S0 * S1 high 32bits of result
S_MUL_HI_U32              SOP2            No              D = S0 * S1 high 32bits of result
S_PACK_LL_B32_B16         SOP2            No              D = { S1[15:0], S0[15:0] }
S_PACK_LH_B32_B16         SOP2            No              D = { S1[31:16], S0[15:0] }
S_PACK_HL_B32_B16         SOP2            No              D = { S1[15:0], S0[31:16] }
