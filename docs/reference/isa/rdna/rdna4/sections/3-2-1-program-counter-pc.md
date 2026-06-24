# 3.2.1. Program Counter (PC)

> RDNA4 ISA — pages 24–24

Abbrev.               Name                       Size     Description
                                                 (bits)
EXCP_FLAG_USER        User-clearable Exception   32       Mask of exceptions that have occurred. Writable by user
                      Flags                               shader.
TBA                   Trap Base Address          48       Holds the program address of the trap handler. Per-VMID
                                                          register. Bit [63] indicates if the trap handler is present (1) or
                                                          not (0) and is not considered part of the address (bit[62] is
                                                          replicated into address bit[63] to form a 64-bit address).
                                                          Accessed via S_SENDMSG_RTN.
TMA                   Trap Memory Address        48       Temporary register for shader operations. For example, can
                                                          hold a pointer to data memory used by the trap handler.
TTMP0-TTMP15          Trap Temporary SGPRs       32       16 SGPRs available only to the Trap Handler for temporary
                                                          storage.
LOADcnt               Vector memory load         6        Counts the number of VMEM load (and atomic with return
                      instruction count                   value) instructions issued but not yet completed.
STOREcnt              Vector memory store        6        Counts the number of VMEM store (and atomic without
                      instruction count                   return) instructions issued but not yet completed.
DScnt                 LDS instruction count      6        Counts the number of LDS instructions issued but not yet
                                                          completed.
KMcnt                 Constant and Message count 5        Counts the number of constant-fetch (scalar memory read),
                                                          and message instructions issued but not yet completed.
SAMPLEcnt             Vector memory sample       6        Counts the number of VMEM sample/gather/msaa-load/get-
                      instruction count                   lod instructions (VSAMPLE encoding) issued but not yet
                                                          completed.
BVHcnt                Vector memory BVH          3        Counts the number of VMEM BVH instructions issued but
                      instruction count                   not yet completed.
EXPcnt                Export Count               3        Counts the number of Export instructions issued but not yet
                                                          completed. Also counts parameter loads outstanding.

3.2. Control State: PC and EXEC

3.2.1. Program Counter (PC)
The Program Counter is a DWORD-aligned byte address that points to the next instruction to execute. When a
wave is created the PC is initialized to the first instruction in the program. The address is DWORD-aligned so the
two LSBs are forced to zero.

There are a few instructions that interact directly with the PC: S_GETPC_B64, S_SETPC_B64, S_CALL_B64,
S_RFE_B64, and S_SWAPPC_B64. These transfer the PC to and from an even-aligned SGPR pair (zero-
extended).

Branches jump to (PC_of_the_instruction_after_the_branch + offset*4). Branches, GET_PC and SWAP_PC are PC-
relative to the next instruction, not the current one. S_TRAP, on the other hand, saves the PC of the S_TRAP
instruction itself.

During wave debugging the program counter may be read. The PC points to the next instruction to issue. All
prior instructions have been issued but may or may not have completed execution.
