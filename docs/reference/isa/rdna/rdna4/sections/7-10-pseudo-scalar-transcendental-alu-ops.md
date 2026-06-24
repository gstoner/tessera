# 7.10. Pseudo-scalar Transcendental ALU ops

> RDNA4 ISA — pages 98–98

When using DPP8 with VOP3/VOP3P, the OPSEL field must be set such that the low result only uses low inputs,
and the high result only uses high inputs. DPP8 follows DPP16’s "BC = 1" behavior and assumes all source lanes
are in-range.

DPP8 Instruction Fields

Field                Size       Description
SRC                  8          Source 0 (VGPR). Since the VOP1/VOP2 source0 slot was filled with the constant "DPP" or
                                "DPPFI", this field provides the actual source0 vgpr.
SEL0                 3          Selects which lane to pull data from, within a group of 8 lanes.
SEL1                            SEL0 selects which lane to read from to supply data into lane 0.
SEL2                            SEL1 selects which lane to read from to supply data into lane 1.
SEL3                            etc.
SEL4                            0 = read from lane 0, 1 = read from lane 1, … 7 = read from lane 7.
SEL5                            Lanes 0-7 can pull from any of lanes 0-7; lanes 8-15 can pull from lanes 8-15, etc.
SEL6
SEL7

7.10. Pseudo-scalar Transcendental ALU ops
This is a collection of VALU ops that operate on a single lane of data where both the source and destination are
SGPRs. These use the VALU pipeline like any other VALU op, and are encoded in the VOP3 format.

                  Pseudo-Scalar F32 Trans ops                  Pseudo-Scalar F16 Trans ops
                  V_S_EXP_F32                                  V_S_EXP_F16
                  V_S_LOG_F32                                  V_S_LOG_F16
                  V_S_RCP_F32                                  V_S_RCP_F16
                  V_S_RSQ_F32                                  V_S_RSQ_F16
                  V_S_SQRT_F32                                 V_S_SQRT_F16

Notes
  • Half-SGPRs are not supported for 16-bit data. The data is expected to be in bits[15:0], and the full 32-bits are
    written with the upper half receiving zeros.
  • These use the usual DENORMAL and ROUND mode bits
  • These produce exceptions like their VALU equivalent instructions
  • Value of EXEC is ignored and the instructions execute even when EXEC==0.
  • VCC may not be used as a destination
  • OPSEL[3] must be set to zero (write to LSBs of SGPR); setting to 1 may result in garbage data.
  • The destination is specified in VDST, not in the SDST field.

7.11. VGPR Indexing
The VALU provides a set of instructions that move or swap VGPRs where the source, dest or both are indexed
by a value in the M0 register. Indices are unsigned.

                                         Table 40. VGPR Indexing Instructions
