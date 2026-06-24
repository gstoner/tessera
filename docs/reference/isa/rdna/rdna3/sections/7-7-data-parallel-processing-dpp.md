# 7.7. Data Parallel Processing (DPP)

> RDNA3 ISA — pages 78–78

VOPD Instruction Fields

Field          Size     Description
opX            4        instruction opcode for the X operation
opY            5        instruction opcode for the Y operation
src0X          9        Source 0 for X operation. May be a VGPR, SGPR, exec, inline or literal constant
src0Y          9        Source 0 for Y operation. May be a VGPR, SGPR, exec, inline or literal constant
vsrc1X         8        Source 1 for X operation. Must be a VGPR. Ignored for V_MOV_B32
vsrc1Y         8        Source 1 for Y operation. Must be a VGPR. Ignored for V_MOV_B32
vdstX          8        Destination VGPR for X operation.
vdstY          7        Destination VGPR for Y operation. vdstY specifies bits [7:1]. The LSB of the destination address is:
                        !vdstX[0]. vdstX and vdstY: one must be even and the other is an odd VGPR.

See VOPD for a list of opcodes usable in the X and Y opcode fields.

V_CNDMASK_B32 is the "VOP2" form that uses VCC as the select. VCC counts as one SGPR read.

VOPD instruction pairs generate only a single exception if either or both raise an exception.

7.7. Data Parallel Processing (DPP)
Data Parallel Processing (DPP) operations allow VALU instruction to select operands from different lanes
(threads) rather than just using a thread’s own data. DPP operations are indicated by the use of the inline
constant: DPP8 or DPP16 in the SRC0 operand. Note that since SRC0 is set to the DPP value, the actual VGPR
address for SRC0 comes from the DPP DWORD.

One example of using DPP is for scan operations. A scan operation is one that computes a value per thread that
is based on the values of the previous threads and possibly itself. E.g. a running sum is the sum of the values
from previous threads in the vector. A reduction operation is essentially a scan that returns a single value from
the highest numbered active thread. A scan operation requires that the EXEC mask to be set to all 1’s for proper
operation. Unused threads (lanes) should be set to a value that does not change the result prior to the scan.

There are two forms of the DPP instruction word:

  DPP8        allows arbitrary swizzling between groups of 8 lanes

  DPP16       allows a set of predefined swizzles between groups of 16 lanes

DPP may be used only with: VOP1, VOP2, VOPC, VOP3 and VOP3P (but not "packed math" ops).
DPP instructions incur an extra cycle of delay to execute.

                                        Table 30. Which instructions support DPP
