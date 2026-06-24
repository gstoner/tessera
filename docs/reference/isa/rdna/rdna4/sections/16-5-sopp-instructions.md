# 16.5. SOPP Instructions

> RDNA4 ISA — pages 276–288

16.5. SOPP Instructions

S_NOP                                                                                                             0

Do nothing. Delay issue of next instruction by a small, fixed amount.

Insert 0..15 wait states based on SIMM16[3:0]. 0x0 means the next instruction can issue on the next clock, 0xf
means the next instruction can issue 16 clocks later.

  for i in 0U : SIMM16.u16[3 : 0].u32 do
        nop()
  endfor

Notes

This instruction may be used to introduce wait states to resolve hazards; see the shader programming guide for
details. Compare with S_SLEEP.

Examples:

        s_nop 0           // Wait 1 cycle.
        s_nop 0xf         // Wait 16 cycles.

S_SETKILL                                                                                                         1

Kill this wave if the least significant bit of the immediate constant is 1.

Used primarily for debugging kill wave host command behavior.

S_SETHALT                                                                                                         2

Set or clear the HALT or FATAL_HALT status bits.

The particular status bit is chosen by halt type control as indicated in SIMM16[2]; 0 = HALT bit select; 1 =
FATAL_HALT bit select.

When halt type control is set to 0 (HALT bit select): Set HALT bit to value of SIMM16[0]; 1 = halt, 0 = clear HALT
bit. The halt flag is ignored while PRIV == 1 (inside trap handlers) but the shader halts after the handler returns
if HALT is still set at that time.

When halt type control is set to 1 (FATAL HALT bit select): Set FATAL_HALT bit to value of SIMM16[0]; 1 =

fatal_halt, 0 = clear FATAL_HALT bit. Setting the fatal_halt flag halts the shader in or outside of the trap
handlers.

S_SLEEP                                                                                                           3

Cause a wave to sleep for up to ~8000 clocks, or to sleep until an external event wakes the wave up.

SIMM16[6:0] determines the sleep duration. The wave sleeps for (64*(SIMM16[6:0]-1) … 64*SIMM16[6:0])
clocks. The exact amount of delay is approximate. Compare with S_NOP. When SIMM16[6:0] is zero then no
sleep occurs.

"Sleep Forever" mode may also be enabled to cause a wave to sleep until an external event wakes it up.

The SIMM16 argument is encoded as follows:

DURATION = SIMM16[6:0]
   Determines the sleep duration. The wave sleeps for (64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0]) clocks. The
   exact amount of delay is approximate. Compare with S_NOP. When set to zero, no sleep occurs.

SLEEP_FOREVER = SIMM16[15]
   If set to 1, enables "sleep forever" mode. The wave sleeps until woken up by one of the following:

     • S_WAKEUP
     • S_WAKEUP_BARRIER
     • An exception
     • Host trap
     • Other trap events
     • Wave KILL

See also S_SLEEP_VAR.

Notes

Examples:

        s_sleep { duration: 0 }        // Wait for 0 clocks.
        s_sleep { duration: 1 }        // Wait for 1-64 clocks.
        s_sleep { duration: 2 }        // Wait for 65-128 clocks.
        s_sleep { sleep_forever: 1 }        // Wait until an event occurs.

S_CLAUSE                                                                                                          5

Mark the beginning of a clause.

The next instruction determines the clause type, which may be one of the following types.

  • Image Load (non-sample instructions )

  • Image Sample
  • Image Store
  • Image Atomic
  • Buffer/Global/Scratch Load
  • Buffer/Global/Scratch Store
  • Buffer/Global/Scratch Atomic
  • Flat Load
  • Flat Store
  • Flat Atomic
  • LDS (loads, stores, atomics may be in same clause)
  • Scalar Memory
  • Vector ALU

Once the clause type is determined, any instruction encountered within the clause that is not of the same type
(and not an internal instruction described below) is illegal and may lead to undefined behaviour. Attempting to
issue S_CLAUSE while inside a clause is also illegal.

Instructions that are processed internally do not interrupt the clause. The following instructions are internal:

  • S_NOP,
  • S_WAIT_*CNT,
  • S_DELAY_ALU.

Halting or killing a wave breaks the clause. VALU exceptions and other traps that cause the shader to enter its
trap handler breaks the clause. The single-step debug mode breaks the clause.

The clause length must be between 2 and 63 instructions, inclusive. Clause breaks may be from 1 to 15, or may
be disabled entirely. Clause length and breaks are encoded in the SIMM16 argument as follows:

LENGTH = SIMM16[5:0]
   This field is set to the logical number of instructions in the clause, minus 1 (e.g. if a clause has 4
   instructions, program this field to 3). The minimum number of instructions required for a clause is 2 and
   the maximum number of instructions is 63, therefore this field must be programmed in the range [1, 62]
   inclusive.

BREAK_SPAN = SIMM16[11:8]
   This field is set to the number of instructions to issue before each clause break. If set to zero then there are
   no clause breaks. If set to nonzero value then the maximum number of instructions between clause breaks
   is 15.

The following instruction types cannot appear in a clause:

  • Instructions of a different type than the clause type, as determined by the first instruction in the clause
  • S_CLAUSE
  • S_ENDPGM
  • SALU
  • Branch
  • Message
  • S_SLEEP

  • S_SLEEP_VAR
  • VDSDIR
  • VINTERP
  • Export
  • S_SETHALT
  • S_SETKILL

To schedule an S_WAIT_* or S_DELAY_ALU instruction for the first instruction in the clause, the waitcnt/delay
instruction must appear before the S_CLAUSE instruction so that S_CLAUSE can accurately determine the
clause type.

S_DELAY_ALU is orthogonal to S_CLAUSE; ALU clauses should be structured to avoid any stalling.

S_DELAY_ALU                                                                                                 7

Insert delay between dependent SALU/VALU instructions.

The SIMM16 argument is encoded as:

INSTID0 = SIMM16[3:0]
   Hazard to delay for with the next VALU instruction.

INSTSKIP = SIMM16[6:4]
   Identify the VALU instruction that the second delay condition applies to.

INSTID1 = SIMM16[10:7]
   Hazard to delay for with the VALU instruction identified by INSTSKIP.

Legal values for the InstID0 and InstID1 fields are:

INSTID_NO_DEP (0x0)
   No dependency on any prior instruction.

INSTID_VALU_DEP_1 (0x1)
   Dependent on previous VALU instruction, 1 instruction(s) back.

INSTID_VALU_DEP_2 (0x2)
   Dependent on previous VALU instruction, 2 instruction(s) back.

INSTID_VALU_DEP_3 (0x3)
   Dependent on previous VALU instruction, 3 instruction(s) back.

INSTID_VALU_DEP_4 (0x4)
   Dependent on previous VALU instruction, 4 instruction(s) back.

INSTID_TRANS32_DEP_1 (0x5)
   Dependent on previous TRANS32 instruction, 1 instruction(s) back.

INSTID_TRANS32_DEP_2 (0x6)
   Dependent on previous TRANS32 instruction, 2 instruction(s) back.

INSTID_TRANS32_DEP_3 (0x7)
   Dependent on previous TRANS32 instruction, 3 instruction(s) back.

INSTID_FMA_ACCUM_CYCLE_1 (0x8)
   Single cycle penalty for FMA accumulation (reserved).

INSTID_SALU_CYCLE_1 (0x9)
   1 cycle penalty for a prior SALU instruction.

INSTID_SALU_CYCLE_2 (0xa)
   2 cycle penalty for a prior SALU instruction.

INSTID_SALU_CYCLE_3 (0xb)
   3 cycle penalty for a prior SALU instruction.

Legal values for the InstSkip field are:

INSTSKIP_SAME (0x0)
   Apply second dependency to same instruction (2 dependencies on one instruction).

INSTSKIP_NEXT (0x1)
   Apply second dependency to next instruction (no skip).

INSTSKIP_SKIP_1 (0x2)
   Skip 1 instruction(s) then apply dependency.

INSTSKIP_SKIP_2 (0x3)
   Skip 2 instruction(s) then apply dependency.

INSTSKIP_SKIP_3 (0x4)
   Skip 3 instruction(s) then apply dependency.

INSTSKIP_SKIP_4 (0x5)
   Skip 4 instruction(s) then apply dependency.

This instruction describes dependencies for two instructions, directing the hardware to insert delay if the
dependent instruction was issued too recently to forward data to the second.

S_DELAY_ALU instructions record the required delay with respect to a previous VALU instruction and indicate
data dependencies that benefit from having extra idle cycles inserted between them. These instructions are
optional: without them the program still functions correctly but performance may suffer when multiple waves
are in flight; IB may issue dependent instructions that stall in the ALU, preventing those cycles from being
utilized by other wavefronts.

If enough independent instructions are between dependent ones then no delay is necessary and this
instruction may be omitted. For wave64 the compiler may not know the status of the EXEC mask and hence
does not know if instructions require 1 or 2 passes to issue. S_DELAY_ALU encodes the type of dependency so
that hardware may apply the correct delay depending on the number of active passes.

S_DELAY_ALU may execute in zero cycles.

To reduce instruction stream overhead the S_DELAY_ALU instructions packs two delay values into one
instruction, with a "skip" indicator so the two delayed instructions don't need to be back-to-back.

S_DELAY_ALU is illegal inside of a clause created by S_CLAUSE.

Example:

  v_mov_b32 v3, v0
  v_lshlrev_b32     v30, 1, v31
  v_lshlrev_b32     v24, 1, v25
  s_delay_alu { instid0: INSTID_VALU_DEP_3, instskip: INSTSKIP_SKIP_1, instid1: INSTID_VALU_DEP_1 }
      // 1 cycle delay here
  v_add_f32   v0, v1, v3
  v_sub_f32   v11, v9, v9
      // 2 cycles delay here
  v_mul_f32   v10, v13, v11

S_WAITCNT                                                                                                             9

Equivalent to S_WAIT_IDLE. This opcode should not be used in modern code; use one of the specialized
S_WAIT_* instructions instead. The operand is ignored for compatibility.

S_WAIT_IDLE                                                                                                          10

Wait for all activity in the wave to be complete (all dependency and memory counters at zero).

S_WAIT_EVENT                                                                                                         11

Wait for an event to occur or a condition to be satisfied before continuing. The SIMM16 argument specifies
which event(s) to wait on.

EXPORT_READY = SIMM16[1]
   If this value is ONE then sleep until the export_ready bit is 1. If the export_ready bit is already 1, no sleep
   occurs. Effect is the same as the export_ready check performed before issuing an export instruction.

No wait occurs if this value is ZERO.

This wait cannot be preempted by KILL, context-save, host trap, single-step or trap after instruction events. IB
waits for the event to occur before processing internal or external exceptions which can delay entry to the trap
handler for a significant amount of time.

S_TRAP                                                                                                               16

Enter the trap handler.

This instruction may be generated internally as well in response to a host trap (TrapID = 0) or an exception.
TrapID 0 is reserved for hardware use and should not be used in a shader-generated trap.

  TrapID = SIMM16.u16[3 : 0].u8;
  "Wait for all instructions to complete";
  // PC passed into trap handler points to S_TRAP itself,
  // *not* to the next instruction.
  { TTMP[1], TTMP[0] } = { TrapID[3 : 0], 12'0, PC[47 : 0] };
  PC = TBA.i64;
  // trap base address
  WAVE_STATUS.PRIV = 1'1U

S_ROUND_MODE                                                                                                    17

Set floating point round mode using an immediate constant.

Avoids wait state penalty that would be imposed by S_SETREG.

S_DENORM_MODE                                                                                                   18

Set floating point denormal mode using an immediate constant.

Avoids wait state penalty that would be imposed by S_SETREG.

S_BARRIER_WAIT                                                                                                  20

Wait for a barrier to complete. The SIMM16 argument specifies which barrier to wait on.

  ;

  // barrierBit 0: reserved
  // barrierBit 1: workgroup
  // barrierBit 2: trap
  barrierBit = SIMM16.i32 >= 0 ? 0 : SIMM16.i32 == -1 ? 1 : 2;
  while !WAVE_BARRIER_COMPLETE[barrierBit] do
      // Implemented as a power-saving idle
      s_nop(16'0U)
  endwhile;
  WAVE_BARRIER_COMPLETE[barrierBit] = 1'0U

S_CODE_END                                                                                                      31

Generate an illegal instruction interrupt. This instruction is used to mark the end of a shader buffer for debug
tools.

This instruction should not appear in typical shader code. It is used to pad the end of a shader program to make
it easier for analysis programs to locate the end of a shader program buffer. Use of this opcode in an embedded
shader block may cause analysis tools to fail.

To unambiguously mark the end of a shader buffer, this instruction must be specified five times in a row (total
of 20 bytes) and analysis tools must ensure the opcode occurs at least five times to be certain they are at the end
of the buffer. This is because the bit pattern generated by this opcode could incidentally appear in a valid
instruction's second dword, literal constant or as part of a multi-DWORD image instruction.

In short: do not embed this opcode in the middle of a valid shader program. DO use this opcode 5 times at the
end of a shader program to clearly mark the end of the program.

Example:

        ...
        s_endpgm       // last real instruction in shader buffer
        s_code_end        // 1
        s_code_end        // 2
        s_code_end        // 3
        s_code_end        // 4
        s_code_end        // done!

S_BRANCH                                                                                                        32

Jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  PC = PC + signext(SIMM16.i16 * 16'4) + 4LL;
  // short jump.

Notes

For a long jump or an indirect jump use S_SETPC_B64.

Examples:

        s_branch label      // Set SIMM16 = +4 = 0x0004
        s_nop 0      // 4 bytes
  label:
        s_nop 0      // 4 bytes
        s_branch label      // Set SIMM16 = -8 = 0xfff8

S_CBRANCH_SCC0                                                                                     33

If SCC is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if SCC == 1'0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_CBRANCH_SCC1                                                                                     34

If SCC is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if SCC == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_CBRANCH_VCCZ                                                                                     35

If VCCZ is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if VCCZ.u1 == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_CBRANCH_VCCNZ                                                                                    36

If VCCZ is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if VCCZ.u1 == 1'0U then

      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_CBRANCH_EXECZ                                                                                             37

If EXECZ is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if EXECZ.u1 == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_CBRANCH_EXECNZ                                                                                            38

If EXECZ is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

  if EXECZ.u1 == 1'0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif

S_ENDPGM                                                                                                    48

End of program; terminate wavefront.

The hardware implicitly executes S_WAIT_IDLE before executing this instruction. See S_ENDPGM_SAVED for
the context-switch version of this instruction.

S_ENDPGM_SAVED                                                                                              49

End of program; signal that a wave has been saved by the context-switch trap handler and terminate
wavefront.

The hardware implicitly executes S_WAIT_IDLE before executing this instruction. See S_ENDPGM for

additional variants.

S_WAKEUP                                                                                                         52

Allow a wave to 'ping' all the other waves in its threadgroup to force them to wake up early from an S_SLEEP
instruction.

The ping is ignored if the waves are not sleeping. This allows for efficient polling on a memory location. The
waves which are polling can sit in a long S_SLEEP between memory reads, but the wave which writes the value
can tell them all to wake up early now that the data is available. This method is also safe from races since any
waves that miss the ping resume when they complete their S_SLEEP.

If the wave executing S_WAKEUP is in a threadgroup (in_wg set), then it wakes up all waves associated with the
same threadgroup ID. Otherwise, S_WAKEUP is treated as an S_NOP.

S_SETPRIO                                                                                                        53

Change wave user priority.

User settable wave priority is set to SIMM16[1:0]. 0 is the lowest priority and 3 is the highest. The overall wave
priority is:

  SysUserPrio = MIN(3, SysPrio[1:0] + UserPrio[1:0]). Priority = {SysUserPrio[1:0], WaveAge[3:0]}

The system priority cannot be modified from within the wave.

S_SENDMSG                                                                                                        54

Send a message to upstream control hardware.

SIMM16[7:0] contains the message type.

Notes

Message types are documented in the shader programming guide.

S_SENDMSGHALT                                                                                                    55

Send a message to upstream control hardware and then HALT the wavefront; see S_SENDMSG for details.

S_INCPERFLEVEL                                                                                                   56

Increment performance counter specified in SIMM16[3:0] by 1.

S_DECPERFLEVEL                                                        57

Decrement performance counter specified in SIMM16[3:0] by 1.

S_ICACHE_INV                                                          60

Invalidate entire first level instruction cache.

S_WAIT_LOADCNT                                                        64

Wait until LOADCNT is less than or equal to SIMM16[5:0].

S_WAIT_STORECNT                                                       65

Wait until STORECNT is less than or equal to SIMM16[5:0].

S_WAIT_SAMPLECNT                                                      66

Wait until SAMPLECNT is less than or equal to SIMM16[5:0].

S_WAIT_BVHCNT                                                         67

Wait until BVHCNT is less than or equal to SIMM16[2:0].

S_WAIT_EXPCNT                                                         68

Wait until EXPCNT is less than or equal to SIMM16[2:0].

S_WAIT_DSCNT                                                          70

Wait until DSCNT is less than or equal to SIMM16[5:0].

S_WAIT_KMCNT                                                          71

Wait until KMCNT is less than or equal to SIMM16[4:0].

S_WAIT_LOADCNT_DSCNT                                                                                        72

Wait until LOADCNT is less than or equal to SIMM16[13:8] and DSCNT is less than or equal to SIMM16[5:0].

Argument is a bitfield of which dependency counters to wait to be zero. The SIMM16 argument is encoded as:

DS = SIMM16[5:0]
   Wait for DSCNT <= N.

MEM = SIMM16[13:8]
   Wait for either LOADCNT <= N or STORECNT <= N (depending on instruction).

S_WAIT_STORECNT_DSCNT                                                                                       73

Wait until STORECNT is less than or equal to SIMM16[13:8] and DSCNT is less than or equal to SIMM16[5:0].

Argument is a bitfield of which dependency counters to wait to be zero. The SIMM16 argument is encoded as:

DS = SIMM16[5:0]
   Wait for DSCNT <= N.

MEM = SIMM16[13:8]
   Wait for either LOADCNT <= N or STORECNT <= N (depending on instruction).
