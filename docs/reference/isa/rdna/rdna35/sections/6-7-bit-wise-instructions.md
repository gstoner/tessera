# 6.7. Bit-Wise Instructions

> RDNA3.5 ISA — pages 60–61

Instruction               Encoding          Sets SCC?        Operation
S_PACK_HH_B32_B16         SOP2              No               D = { S1[31:16], S0[31:16] }

6.5. Conditional Move Instructions
Conditional instructions use the SCC flag to determine whether to perform the operation, or (for CSELECT)
which source operand to use.

                       Table 22. Conditional Instructions
Instruction               Encoding Sets SCC? Operation
S_CSELECT_{B32, B64}      SOP2        No             D = SCC ? S0 : S1.
S_CMOVK_I32               SOPK        No             if (SCC) D = signext(simm16).
S_CMOV_{B32,B64}          SOP1        No             if (SCC) D = S0, else NOP.

6.6. Comparison Instructions
These instructions compare two values and set the SCC to 1 if the comparison yielded a TRUE result.

                                             Table 23. Conditional Instructions
Instruction                          Encoding         Sets SCC?     Operation
S_CMP_EQ_U64, S_CMP_LG_U64           SOPC             Test          Compare two 64-bit source values. SCC = S0 <cond> S1.
S_CMP_{EQ,LG,GT,GE,LE,LT}_{I32 SOPC                   Test          Compare two source values. SCC = S0 <cond> S1.
,U32}
S_BITCMP0_{B32,B64}                  SOPC             Test          Test for "is a bit zero". SCC = !S0[S1].
S_BITCMP1_{B32,B64}                  SOPC             Test          Test for "is a bit one". SCC = S0[S1].

6.7. Bit-Wise Instructions
Bit-wise instructions operate on 32- or 64-bit data without interpreting it has having a type. For bit-wise
operations if noted in the table below, SCC is set if the result is nonzero.

                                                 Table 24. Bit-Wise Instructions
Instruction                                  Encoding        Sets SCC? Operation
S_MOV_{B32,B64}                              SOP1            No           D = S0
S_MOVK_I32                                   SOPK            No           D = signext(simm16)
{S_AND,S_OR,S_XOR}_{B32,B64}                 SOP2            D!=0         D = S0 & S1, S0 OR S1, S0 XOR S1
{S_AND_NOT1,S_OR_NOT1}_{B32,B64}             SOP2            D!=0         D = S0 & ~S1, S0 OR ~S1
{S_NAND,S_NOR,S_XNOR}_{B32,B64}              SOP2            D!=0         D = ~(S0 & S1), ~(S0 OR S1), ~(S0 XOR S1)
S_LSHL_{B32,B64}                             SOP2            D!=0         D = S0 << S1[4:0], [5:0] for B64.
S_LSHR_{B32,B64}                             SOP2            D!=0         D = S0 >> S1[4:0], [5:0] for B64.
S_ASHR_{I32,I64}                             SOP2            D!=0         D = sext(S0 >> S1[4:0]) ([5:0] for I64).
S_BFM_{B32,B64}                              SOP2            No           Bit field mask
                                                                          D = ( (1 << S0[4:0]) -1) << S1[4:0]
                                                                          (uses [5:0] for the B64 version)

Instruction                              Encoding   Sets SCC? Operation
S_BFE_U32, S_BFE_U64                     SOP2       D!=0     Bit Field Extract, then sign extend result for I32/64
S_BFE_I32, S_BFE_I64                                         instructions.
(signed/unsigned)                                            S0 = data, S1[22:16]= width
                                                             I32/U32: S1[4:0] = offset
                                                             I64/U64: S1[5:0] = offset
S_NOT_{B32,B64}                          SOP1       D!=0     D = ~S0.
S_WQM_{B32,B64}                          SOP1       D!=0     D = wholeQuadMode(S0)
                                                             Per quad (4 bits): set the result to 1111 if any of the 4
                                                             bits in the corresponding source mask are set to 1.
                                                             D[n*4] = (S[n*4] || S[n*4+1] || S[n*4+2] || S[n*4+3] )
                                                             D[n*4+1] = (S[n*4] || S[n*4+1] || S[n*4+2] || S[n*4+3] )
                                                             D[n*4+2] = (S[n*4] || S[n*4+1] || S[n*4+2] || S[n*4+3] )
                                                             D[n*4+3] = (S[n*4] || S[n*4+1] || S[n*4+2] || S[n*4+3] )
S_QUADMASK_{B32,B64}                     SOP1       D!=0     Create a 1-bit per quad mask from a 1 bit per pixel
                                                             mask.
                                                             Creates an 8-bit mask from 32-bits, or 16 bits from 64.
                                                             D[0] = (S0[3:0] != 0),
                                                             D[1] = (S0[7:4] != 0), …
S_BITREPLICATE_B64_B32                   SOP1       No       Replicate each bit in 32-bit S0 twice:
                                                             D = { … S0[1], S0[1], S0[0], S0[0] }.
                                                             Two of these instructions is the inverse of
                                                             S_QUADMASK.
                                                             Two of these instructions expands a quad mask into a
                                                             thread-mask.
S_BREV_{B32,B64}                         SOP1       No       D = S0[0:31] are reverse bits.
S_BCNT0_I32_{B32,B64}                    SOP1       D!=0     D = CountZeroBits(S0).
S_BCNT1_I32_{B32,B64}                    SOP1       D!=0     D = CountOneBits(S0).
S_CTZ_I32_{B32,B64}                      SOP1       No       Count Trailing zeroes: Find-first One from LSB.
                                                             D = Bit position of first one in S0
                                                             starting from LSB. -1 if not found
S_CLZ_I32_{B32,B64}                      SOP1       No       Count Leading zeroes. D = "how many zeros before
                                                             the first one starting from the MSB".
                                                             Returns -1 if none.
S_CLS_I32_{B32,B64}                      SOP1       N        Count Leading Sign-bits: Count how many bits in a
                                                             row (from MSB to LSB) are the same as the sign bit.
                                                             Return -1 if the input is zero or all 1’s (-1). 32-bit
                                                             pseudo-code:

                                                                if (S0 == 0 || S0 == -1) D = -1
                                                                else
                                                                    D = 0
                                                                    for (I = 31 .. 0)
                                                                        if (S0[I] == S0[31])
                                                                           D++
                                                                        else break

S_BITSET0_{B32,B64}                      SOP1       No       D[S0[4:0], [5:0] for B64] = 0
S_BITSET1_{B32,B64}                      SOP1       No       D[S0[4:0], [5:0] for B64] = 1
