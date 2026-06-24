# 7.2. Operands

> RDNA4 ISA — pages 78–78

Field     Size     Description
CM        1        clamp or compare-signal (depends on opcode):
                   V_CMP: CM=1 means signaling-compare when qNaN detected; 0 = non-signaling
                   Float arithmetic: clamp result to [0, 1.0]; -0 is clamped to +0.
                   Signed integer arithmetic: clamp result to [INT_MIN, INT_MAX]
                   Unsigned integer arithmetic: clamp result to [0, +UINT_MAX]
                   Where "INT_MIN" and "INT_MAX" are the largest negative and positive representable integers for the
                   size of integer being used (16, 32 or 64 bit). "UINT_MAX" is the largest unsigned int.
OPSEL     4        Operation select for 16-bit math: 1=select high half, 0=select low half
                   [0]=src0, [1]=src1, [2]=src2, [3]=dest
                   For dest=0, dest_vgpr[31:0] = {prev_dst_vgpr[31:16], result[15:0] }
                   For dest=1, dest_vgpr[31:0] = {result[15:0], prev_dst_vgpr[15:0] }
                   OPSEL may only be used for 16-bit operands, and must be zero for any other operands/results.

                   Uniquely for V_PERMLANE*: OPSEL[0] is "fetch invalid"; OPSEL[1] is "bounds control" (like DPP8).
                   DOT2_F16 and_BF16: src0 and src1 must have OPSEL[1:0] = 0.
                   OPSEL may be used to select upper or lower halves of an SGPR data input.
                   Certain operations described in later sections overload OPSEL to have a different meaning.

7.2. Operands
Most VALU instructions take at least one input operand. The data-size of the operands is explicitly defined in
the name of the instruction. For example, V_FMA_F32 operates on 32-bit floating point data.

VGPR Alignment: there is no alignment restriction for single or double-float operations.

                                        Table 34. VALU Instruction Operands
