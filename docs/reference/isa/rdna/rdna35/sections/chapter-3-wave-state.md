# Chapter 3. Wave State

> RDNA3.5 ISA — pages 22–22

Chapter 3. Wave State
This chapter describes the state variables visible to the shader program. Each wave has a private copy of this
state unless otherwise specified.

3.1. State Overview
The table below shows the hardware states readable or writable by a shader program. All registers below are
unique to each wave except for TBA and TMA which are shared.

                                     Table 4. Readable and Writable Hardware States
Abbrev.               Name                         Size     Description
                                                   (bits)
PC                    Program Counter              48       Points to the memory address of the next shader instruction
                                                            to execute. Read/write only via scalar control flow
                                                            instructions and indirectly using branch. The 2 LSB’s are
                                                            forced to zero.
V0-V255               VGPR                         32       Vector general-purpose register. (32 bits per work-item x (32
                                                            or 64) work-items per wave).
S0-S105               SGPR                         32       Scalar general-purpose register. All waves are allocated 106
                                                            SGPRs + 16 TTMPs.
LDS                   Local Data Share             64kB     Local data share is a scratch RAM with built-in arithmetic
                                                            capabilities that allow data to be shared between threads in a
                                                            work-group.
EXEC                  Execute Mask                 64       A bit mask with one bit per thread, which is applied to vector
                                                            instructions and controls which threads execute and which
                                                            ignore the instruction.
EXECZ                 EXEC is zero                 1        A single bit flag indicating that the EXEC mask is all zeros.
                                                            For wave32 it considers only EXEC[31:0].
VCC                   Vector Condition Code        64       A bit mask with one bit per thread; it holds the result of a
                                                            vector compare operation or integer carry-out. Physically
                                                            VCC is stored in specific SGPRs.
VCCZ                  VCC is zero                  1        A single-bit flag indicating that the VCC mask is all zeros. For
                                                            wave32 it considers only VCC[31:0].
SCC                   Scalar Condition Code        1        Result from a scalar ALU comparison instruction.
FLAT_SCRATCH          Flat scratch address         48       The base address of scratch memory for this wave. Used by
                                                            Flat and Scratch instructions. Read-only by user shader.
STATUS                Status                       32       Read-only shader status bits.
MODE                  Mode                         32       Writable shader mode bits.
M0                    Misc Reg                     32       A temporary register that has various uses, including GPR
                                                            indexing and bounds checking.
TRAPSTS               Trap Status                  32       Holds information about exceptions and pending traps.
TBA                   Trap Base Address            48       Holds the pointer to the current trap handler program
                                                            address. Per-VMID register. Bit [63] indicates if the trap
                                                            handler is present (1) or not (0) and is not considered part of
                                                            the address (bit[62] is replicated into address bit[63]).
                                                            Accessed via S_SENDMSG_RTN
TMA                   Trap Memory Address          48       Temporary register for shader operations. For example, can
                                                            hold a pointer to memory used by the trap handler.
