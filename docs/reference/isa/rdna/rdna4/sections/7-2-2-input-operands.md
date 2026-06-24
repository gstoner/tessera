# 7.2.2. Input Operands

> RDNA4 ISA — pages 80–81

Opcode                 Encoding    VDST               SDST        VSRC0         VSRC1                   VSRC2
V_{ADD,SUB,SUBREV} VOP2            add result         n/a         in0           in1                     unused
_CO_U32,                           (VCC=carry-out)                                                      (carry-in=VCC)
V_{ADD,SUB,SUBREV} VOP3SD          add result         carry-out   in0           in1                     carry-in
_CO_CI_U32
V_MAD_*_CO             VOP3SD      result             carry-out   in0           in1                     in2
V_DIV_SCALE            VOP3SD      result             carry-out   in0           in1                     in2
V_READLANE             VOP3        scalar dst (SGPR   n/a         vgpr#         lane-sel: sgpr, M0,     n/a
                                   only)                                        inline
V_READFIRSTLANE        VOP1        scalar dst (SGPR   n/a         vgpr#         n/a (lane-sel = exec)   n/a
                                   only)
V_WRITELANE            VOP3        dest vgpr          n/a         sgpr#, const, lane-sel: sgpr, M0,     n/a
                                                                  M0            inline
V_CMP*                 VOPC        "VCC" implied      n/a         in0           in1                     n/a
                       VOP3SD      cmp-result (sgpr) unused       in0           in1                     unused
V_CNDMASK              VOP2        dest vgpr          n/a         in0           in1                     unused (implied:
                                                                                                        VCC)
                       VOP3        dest vgpr          unused      in0           in1                     select sgpr (e.g.
                                                                                                        VCC)

The readlane lane-select is limited to the valid range of lanes (0-31 for wave32, 0-63 for wave64) by ignoring
upper bits of the lane number.

7.2.2. Input Operands
VALU instructions can use any of the following sources for input, subject to restrictions listed below:
  • VOP1, VOP2, VOPC:
     ◦ SRC0 is 9 bits and may be a VGPR, SGPR (including TTMPs and VCC), M0, EXEC, inline or literal
       constant.
     ◦ VSRC1 is 8 bits and may specify only a VGPR
  • VOP3 : all 3 sources are 9 bits but still have restrictions:
     ◦ Not all VOPC/1/2 instructions are available in VOP3 (only those that benefit from VOP3 encoding).
  • See complete operand list: VALU Instruction Operands

Inline constants with: V_DOT2_F32_BF16, V_DOT2_F32_F16, V_DOT2_F16_F16, and DOT2_BF16_BF16
   For these instructions, the inline constant for sources 0 and 1 replicate the inline constant value into
   bits[31:16]. For source2, the OPSEL bit is used to control replication or not (gets zero if not replicating low
   bits).

7.2.2.1. Input Operand Modifiers
The input modifiers ABS and NEG apply to floating point inputs and are undefined for any other type of input.
In addition, input modifiers are supported for: V_MOV_B32, V_MOV_B16, V_MOVREL*_B32 and V_CNDMASK.
ABS returns the absolute value, and NEG negates the input.

Input modifiers are not supported for:
  • READLANE, READFIRSTLANE, WRITELANE
  • Integer arithmetic or bitwise operations

  • PERMLANE
  • Ops that cannot be promoted to VOP3
  • QSAD
  • SWAP
  • CVT*_F32_{FP8,BF8}
  • DOT2 with F16/BF16 (NEG ok but ABS is not)
  • DOT4 with FP8/BF8 (NEG ok but ABS is not)

7.2.2.2. Source Operand Restrictions
Not every combination of source operands that can be expressed in the microcode format is legal. This section
describes the legal and illegal settings.

   Terminology for this section:
     "scalar value" = SGPR, VCC, M0, EXEC, SCC or literal constant
        and also: shared_base, shared_limit, private_base, private_limit

  • Instructions may use at most two Scalar Values: SGPR, VCC, M0, EXEC, SCC, Literal
  • All instruction formats including VOP3 and VOP3P may use one literal constant
      ◦ Inline constants are free (do not count against 2 scalar value limit).
      ◦ Literals may not be used with DPP
      ◦ It is permissible for both scalar values to be SGPRs, although VCC counts as an SGPR.
            ▪ VCC when used implicitly counts against this limit: ADD*_CI, SUB*_CI, DIV_FMAS, CNDMASK
      ◦ 64-bit shift instructions can use only one scalar value input, and can’t use the same one twice
        (inlines don’t count against this limit)
      ◦ Using the same scalar value twice only counts as a single scalar value, however using the same scalar
        value twice, but with different sizes has specific rules and limits:
          ▪ Using the same literal with different sizes counts as 2 scalar values, not 1.
            ▪ S[0] and S[0:1] can be considered as 1 scalar value, but S[1] and S[0:1] count as 2.
              In general, these rules apply to any S[2n] and S[2n:2n+1] count as one, but S[2n+1] and S[2n:2n+1] count
              as 2.
  • SGPR source rules must be met for both passes of a wave64, bearing in mind that sources that read a mask
    (bit-per-lane) increment the SGPR address for the second pass, and they may not be shared with other
    sources.

7.2.2.3. OPSEL Field Restrictions
The OPSEL field (of VOP3) is usable only for a subset of VOP3 instructions, as well as VOP1/2/C instructions
promoted to VOP3.

                                           Table 35. Opcodes usable with OPSEL
V_ADD_F16                      V_ADD_NC_I16                  V_ADD_NC_U16                 V_CMP*_{I16,U16,F16}
V_SUBREV_F16                   V_SUB_F16                     V_SUB_NC_I16                 V_SUB_NC_U16
V_MAD_I16                      V_MAD_I32_I16                 V_MAD_U16                    V_MAD_U32_U16
V_MUL_F16                      V_MUL_LO_U16                  V_FMAC_F16                   V_FMA_F16
V_ALIGNBIT_B32                 V_ALIGNBYTE_B32                                            V_PACK_B32_F16
V_ASHRREV_I16                  V_ASHR_PK_I8_I32              V_ASHR_PK_U8_I32             V_LSHLREV_B16
