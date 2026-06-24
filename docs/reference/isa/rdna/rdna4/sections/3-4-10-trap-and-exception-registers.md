# 3.4.10. Trap and Exception registers

> RDNA4 ISA — pages 38–39

Field                          Bits    Description
Issued Stalled                 20:14   SIMD was stalled from issuing an instruction of these types during the snapshot
                                       cycle
                                       Same as above, +8.

PERF_SNAPSHOT_DATA2

Field                          Bits    Description
LOADcnt                        5:0     value of wave’s LOADcnt
STOREcnt                       11:6    value of wave’s STOREcnt
BVHcnt                         14:12   value of wave’s BVHcnt
SAMPLEcnt                      20:15   value of wave’s SAMPLEcnt
DScnt                          26:21   value of wave’s DScnt
KMcnt                          31:27   value of wave’s KMcnt

PERF_SNAPSHOT_PC_LO

Field                          Bits    Description
PC_LO                          31:0    Program counter low bits of wave’s PC at the time of snapshot

PERF_SNAPSHOT_PC_HI

Field                          Bits    Description
PC_HI                          15:0    Program counter high bits of wave’s PC at the time of snapshot

3.4.10. Trap and Exception registers
Each type of exception can be enabled or disabled independently by setting, or clearing, bits in the
TRAP_CTRL register. This section describes the registers that control and report shader exceptions.

Trap temporary SGPRs (TTMP*) are privileged for writes - they can be written only when in the trap handler
(STATUS.PRIV = 1). TTMPs can be read by the user shader. When the shader is not privileged
(STATUS.PRIV==0), writes to these are ignored. TMA and TBA are read-only; they can be accessed through
S_SENDMSG_RTN.

When a trap is taken (either user initiated, exception or host initiated), the shader hardware generates an
S_TRAP instruction. TRAP_ID in TTMP1 is zero for exceptions, or the S_TRAP ID for those traps. This loads
trap information into a pair of SGPRs:

  <wait for outstanding instructions to finish>
  { TTMP1, TTMP0 } = {TrapID[3:0], zeros, PC[47:0]} // TrapID=0 unless the exception is "S_TRAP"

STATUS . TRAP_EN

   This bit tells the shader whether or not a trap handler is present. When one is not present, traps are not
   taken no matter whether they’re floating point, user or host-initiated traps. When the trap handler is
   present, the wave may use TTMP0-15 for trap processing.
   If trap_en == 0, all traps and exceptions are ignored except for fatal ones, and S_TRAP is converted by hardware
   to NOP.

TRAP_CTRL

      Exception enable mask. Defines which of the sources of exception cause the shader to jump to the trap
      handler when the exception occurs. 1 = enable traps; 0 = disable traps.
      MEMVIOL and Illegal-Instruction jump to the trap handler and cannot be masked off.

Bit       Exception               Cause                                                             Result
0         alu_invalid             INVALID: operand is invalid for operation: 0 * inf, 0/0, sqrt(-   QNaN
                                  x), any input is SNaN.
1         alu_input_denorm        INPUT DENORMAL: one or more operands was subnormal                ordinary result
2         alu_float_div0          FLOAT DIVIDE BY ZERO: Float X / 0                                 correct signed infinity
3         alu_overflow            OVERFLOW: The rounded result would be larger than the             Depends on rounding mode.
                                  largest finite number.                                            Signed max# or infinity.
4         alu_underflow           UNDERFLOW: The exact or rounded result is less than the           subnormal or zero
                                  smallest normal (non-subnormal) representable number.
5         alu_inexact             INEXACT: The rounded result of a valid operation is different Operation result
                                  from the infinitely precise result.
6         alu_int_div0            INTEGER DIVIDE BY ZERO: Integer X / 0                             undefined
7         addr_watch              ADDRESS WATCH: VMEM or SMEM has witnessed a thread
                                  access an 'address of interest'
8         trap_on_wave_end        Trap before executing S_ENDPGM
9         trap_after_inst         Trap after every instruction (except S_ENDPGM)

EXCP_FLAG_PRIV Register

EXCP_FLAG_PRIV contains flags of which exceptions have occurred. These flags are sticky - they accumulate
(logical OR) exceptions as they occur regardless of TRAP_CTRL settings. This register can be written only in the
trap handler; the user shader may read it. A few of the bits are status bits that do not trigger a trap.

Field                       Bit Pos Description
addr_watch                  3:0     Four bits that Indicate if address watch 0, 1, 2 or 3 have been hit.
memviol                     4       A memory violation has occurred.
save_context                5       A bit set by the host command via GRBM (or context-save/restore unit) indicating that this
                                    wave must jump to its trap handler and save its context. This bit should be cleared by the
                                    trap handler using S_SETREG.
illegal_inst                6       An illegal instruction has been detected. If a trap handler is present and the wave is not in
                                    the trap handler: jump to the trap handler; Otherwise, send an interrupt and halt.
host_trap                   7       Trap handler has been called to service a host trap. Trap may simultaneously have been
                                    called to handle other traps as well.
wave_start                  8       Trap handler has been called before the first instruction of a new wave.
wave_end                    9       Trap handler has been called after the last instruction of a wave.
perf_snapshot               10      Trap handler has been called due to a stochastic performance snapshot
trap_after_inst             11      Trap handler has been called due to "trap after instruction" mode
first_memviol_source 31:30          Indicates the source of the first MEMVIOL: 0=instruction cache, 1=SMEM, 2=LDS,
                                    3=VMEM.

EXCP_FLAG_USER Register

EXCP_FLAG_USER contains flags of which exceptions have occurred. These flags are sticky - they accumulate
(logical OR) exceptions as they occur regardless of TRAP_CTRL settings. This register can be written by the
user shader. A few of the bits are status bits that do not trigger a trap.
