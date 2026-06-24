# 3.2. Control State: PC and EXEC

> RDNA3 ISA — pages 23–23

Abbrev.               Name                     Size     Description
                                               (bits)
TTMP0-TTMP15          Trap Temporary SGPRs     32       16 SGPRs available only to the Trap Handler for temporary
                                                        storage.
VMcnt                 Vector memory load       6        Counts the number of VMEM load and sample instructions
                      instruction count                 issued but not yet completed.
VScnt                 Vector memory store      6        Counts the number of VMEM store instructions issued but
                      instruction count                 not yet completed.
EXPcnt                Export Count             3        Counts the number of Export and GDS instructions issued
                                                        but not yet completed. Also counts parameter loads
                                                        outstanding.
LGKMcnt               LDS, GDS, Constant and   6        Counts the number of LDS, GDS, constant-fetch (scalar
                      Message count                     memory read), and message instructions issued but not yet
                                                        completed.

3.2. Control State: PC and EXEC

3.2.1. Program Counter (PC)
The Program Counter is a DWORD-aligned byte address that points to the next instruction to execute. When a
wave is created the PC is initialized to the first instruction in the program.

There are a few instructions to interact directly with the PC: S_GETPC_B64, S_SETPC_B64, S_CALL_B64,
S_RFE_B64 and S_SWAPPC_B64. These transfer the PC to and from an even-aligned SGPR pair (sign-extended).

Branches jump to (PC_of_the_instruction_after_the_branch + offset*4). Branches, GET_PC and SWAP_PC are PC-
relative to the next instruction, not the current one. S_TRAP, on the other hand, saves the PC of the S_TRAP
instruction itself.

During wave debugging, the program counter may be read. The PC points to the next instruction to issue. All
prior instructions have been issued but may or may not have completed execution.

3.2.2. EXECute Mask
The Execute mask (64-bit) controls which threads in the vector are executed. Each bit indicates how one thread
behaves for vector instructions: 1 = execute, 0 = do not execute. EXEC can be read and written via scalar
instructions, and can also be written as a result of a vector-alu compare. EXEC affects: vector-alu, vector-
memory, LDS, GDS and export instructions. It does not affect scalar execution or branches.

Wave64 uses all 64 bits of the exec mask. Wave32 waves use only bits 31:0 and hardware does not act upon the
upper bits.

There is a summary bit (EXECZ) that indicates that the entire execute mask is zero. It can be used as a condition
for branches to skip code when EXEC is zero. For wave32, this reflects the state of EXEC[31:0].
