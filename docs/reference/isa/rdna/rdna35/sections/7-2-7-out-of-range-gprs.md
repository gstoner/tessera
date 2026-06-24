# 7.2.7. Out-of-Range GPRs

> RDNA3.5 ISA — pages 73–73

7.2.5. Instructions using SGPRs as Mask or Carry
Every VALU instruction can use SGPRs as a constant, but the following can read or write SGPRs as masks or
carry:

Read Mask or Carry in        Write Carry out              Implicitly Reads VCC            Implicitly Writes VCC
V_CNDMASK_B32                V_CMP*                       V_DIV_FMAS_F32                  V_DIV_SCALE_F32
V_ADD_CO_CI_U32              V_ADD_CO_CI_U32              V_DIV_FMAS_F64                  V_DIV_SCALE_F64
V_SUB_CO_CI_U32              V_SUB_CO_CI_U32              (fmas reads 3 operands + VCC)   V_CMP (not V_CMPX)
V_SUBREV_CO_CI_U32           V_SUBREV_CO_CI_U32           V_CNDMASK in VOP2
                             V_ADD_CO_U32                 V_{ADD,SUB,SUBREV}_CO_CI_U
                                                          32 in VOP2
                             V_SUB_CO_U32
                             V_SUBREV_CO_U32
                             V_MAD_U64_U32
                             V_MAD_I64_I32
                             Write Data out (not carry)
                             V_READLANE
                             V_READFIRSTLANE

"VCC" in the above table refers to VCC in a VOP2 or VOPC encoding, or any SGPR specified in the SRC2 or SDST
field for VOP3 encoding, except for DIV_FMAS that implicitly reads VCC (no choice).

V_CMPX is the only VALU instruction that writes EXEC.

7.2.6. Wave64 use of SGPRs
VALU instructions may use SGPRs as a uniform input, shared by all work-items. If the value is used as simple
data value, then the same SGPR is distributed to all 64 work-items. If, on the other hand, the data value
represents a mask (e.g. carry-in, mask for CNDMASK), then each work-item receives a separate value, and two
consecutive SGPRs are read.

7.2.7. Out-of-Range GPRs
When a source VGPR is out-of-range, the instruction uses as input the value from VGPR0.

When the destination GPR is out-of-range, the instruction executes but does not write the results.

See VGPR Out Of Range Behavior for more information.

7.2.8. PERMLANE Specific Rules
V_PERMLANE may not occur immediately after a V_CMPX. To prevent this, any other VALU opcode may be
inserted (e.g. V_NOP).
