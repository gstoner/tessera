# 5.6. Data Dependency Resolution

> RDNA3.5 ISA — pages 53–53

5.6. Data Dependency Resolution
Shader hardware can resolve most data dependencies, but a few cases must be explicitly handled by the shader
program. In these cases, the program must insert S_WAITCNT instructions to ensure that previous operations
have completed before continuing.

The shader has four counters that track the progress of issued instructions. S_WAITCNT waits for the values of
these counters to be at, or below, specified values before continuing. These allow the shader writer to schedule
long-latency instructions, execute unrelated work, and specify when results of long-latency operations are
needed.

Inserting S_NOP is not required to achieve correct operation.

                                         Table 18. Data Dependency Instructions
Instructions                Description
S_WAITCNT                   Wait for count of outstanding instruction counters to be less-than or equal-to all of these
                            values before continuing.
                            SIMM16 = { VMcnt[5:0], LGKMcnt[5:0], 1’b0, EXPcnt[2:0] }
S_WAITCNT_VSCNT             Wait for VSCNT, VMCNT, EXPCNT or LGKMcnt to be less-than or equal-to the count in
S_WAITCNT_LGKMCNT           SIMM16 before continuing.
S_WAITCNT_EXPCNT
S_WAITCNT_VMCNT
S_WAIT_EVENT                Wait for an event to occur before proceeding
                            SIMM16[0] : 1=don’t wait, 0= wait for export-ready; other bits are reserved.
                            Any exception waits for this to complete before being processed, including: KILL, save-
                            context, host trap, memviol and anything that causes a trap to be taken.
S_DELAY_ALU                 Insert delay between dependent SALU/VALU instructions.
                            SIMM16[3:0] = InstID0
                            SIMM16[6:4] = InstSkip
                            SIMM16[10:7] = InstID1
                            This instruction describes dependencies for two instructions, directing the hardware to insert
                            delay if the dependent instruction was issued too recently to forward data to the second. For
                            details, see: S_DELAY_ALU.

S_WAITCNT* waits for outstanding instructions that use the specified counter to complete. Instructions within
a type often return in the order they were issued compared to other instructions of that type, but typically
return out of order with respect to instructions of different types. These counters count instructions, not threads.

These are the memory instruction groups - each returns out of order with respect to the others:
  • VMcnt:
     ◦ Texture SAMPLE
      ◦ Texture/Buffer/Global/Scratch/Flat Loads and atomic-with-return
  • VScnt:
     ◦ Texture/Buffer/Global/Scratch/Flat Stores and atomic-without-return
  • LGKMcnt:
     ◦ LDS indexed operations
      ◦ SMEM: scalar memory loads may return completely out-of-order with respect to other scalar memory
        loads
      ◦ GDS & GWS
      ◦ FLAT instructions (uses both LGKMcnt and either VMcnt or VScnt)
