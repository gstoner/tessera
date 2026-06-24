# 16.9.1. VOPC using VOP3 encoding

> RDNA4 ISA — pages 398–398

  endif;
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

16.9.1. VOPC using VOP3 encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x000.

When the CLAMP microcode bit is set to 1, these compare instructions signal an exception when either of the
inputs is NaN. When CLAMP is set to zero, NaN does not signal an exception. The second eight VOPC
instructions have {OP8} embedded in them. This refers to each of the compare operations listed below.
