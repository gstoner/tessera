# 3.4.9. Trap and Exception registers

> RDNA3 ISA — pages 34–35

    FLAT_SCRATCH_LO = scratch_base [31:0]
    FLAT_SCRATCH_HI = scratch_base [63:32]

3.4.8. Hardware Internal Registers
These registers are read-only and can be accessed by the S_GETREG instruction. They return information
about hardware allocation and status. HW_ID and the various *_BASE values are not predictable and may
change over the lifetime of a wave if context-switching can occur.

HW_ID1

Field            Bits         Description
WAVE_ID          4:0          Wave id within the SIMD.
SIMD_ID          9:8          SIMD_ID within the WGP: [0] = row, [1] = column.
WGP_ID           13:10        Physical WGP ID.
SA_ID            16           Shader Array ID
SE_ID            20:18        Shader Engine ID
DP_RATE          31:29        Number of double-precision float units per SIMD. 1+log2(#DP-alu’s). 0=none, 1=1/32rate (1 dp
                              lane/clk), 2=1/16 rate (2 dp lanes/clk), 3=1/8, 4=1/4, 5=1/2, 6=full rate (32 dp lanes per clock).

HW_ID2

Field                 Bits         Description
QUEUE_ID              3:0          Queue_ID (also encodes shader stage)
PIPE_ID               5:4          Pipeline ID
ME_ID                 9:8          MicroEngine ID: 0 = graphics, 1 & 2 = ACE compute
STATE_ID              14:12        State context ID
WG_ID                 20:16        Work-group ID (0-31) within the WGP.
VM_ID                 27:24        Virtual Memory ID

Other S_GETREG, S_SETREG targets:

Register                      Bits        Description
FLUSH_IB                      1           Writing this with bit[0]=1 flushes the instruction fetch buffers for the targeted wave.
SH_MEM_BASES                  16, 16      Per-VMID register, readable by the shader, which holds the private and shared
                                          apertures.
PC_LO                         32          Program counter low and high halves. GETREG should not be used to read the PC -
PC_HI                         32          use S_GETPC instead.
FLAT_SCRATCH_HI               32          Flat scratch base address. Only writable when in trap handler
FLAT_SCRATCH_LO               32

Note: TMA and TBA are read using S_SENDMSG_RTN.

3.4.9. Trap and Exception registers
Each type of exception can be enabled or disabled independently by setting, or clearing, bits in the TRAPSTS
register’s EXCP_EN field. This section describes the registers that control and report shader exceptions.

Trap temporary SGPRs (TTMP*) are privileged for writes - they can be written only when in the trap handler

(STATUS.PRIV = 1). TTMPs cannot be read by the user shader (returns zero).

When the shader is not privileged (STATUS.PRIV==0), writes to these are ignored. TMA and TBA are read-only;
they can be accessed through S_SENDMSG_RTN.

When a trap is taken (either user initiated, exception or host initiated), the shader hardware generates an
S_TRAP instruction. This loads trap information into a pair of SGPRS:

  {TTMP1, TTMP0} = {7'h0, HT[0],trapID[7:0], PC[47:0]}.

HT is set to one for host initiated traps, and zero for user traps (s_trap) or exceptions. TRAP_ID is zero for
exceptions, or the user/host trapID for those traps.

STATUS . TRAP_EN

   This bit tells the shader whether or not a trap handler is present. When one is not present, traps are not
   taken no matter whether they’re floating point, user or host-initiated traps. When the trap handler is
   present, the wave uses an extra 16 SGPRs for trap processing.
   If trap_en == 0, all traps and exceptions are ignored, and s_trap is converted by hardware to NOP.

MODE . EXCP_EN[8:0]

   Exception enable mask. Defines which of the sources of exception cause the shader to jump to the trap
   handler when the exception occurs. 1 = enable traps; 0 = disable traps.
   MEMVIOL and Illegal-Instruction jump to the trap handler and cannot be masked off.

   Bit Exception      Cause                                                            Result
   0   invalid        operand is invalid for operation: 0 * inf, 0/0, sqrt(-x), any input QNaN
                      is SNaN.
   1   Input          one or more operands was subnormal                               ordinary result
       Denormal
   2   Divide by zero Float X / 0                                                      correct signed infinity
   3   overflow       The rounded result would be larger than the largest finite       Depends on rounding mode.
                      number.                                                          Signed max# or infinity.
   4   underflow      The exact or rounded result is less than the smallest normal     subnormal or zero
                      (non-subnormal) representable number.
   5   inexact        The rounded result of a valid operation is different from the    Operation result
                      infinitely precise result.
   6   integer divide Integer X / 0                                                    undefined
       by zero
   7   address watch VMEM or SMEM has witnessed a thread access an 'address of
                     interest'
   8   reserved

TRAPSTS Register

TRAPSTS contains information about traps and exceptions, and may be written by user shader or trap handler.
