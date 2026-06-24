# 3.4. Wave State Registers

> RDNA4 ISA — pages 32–32

This memory is private to each thread (although this is not enforced) and memory swizzling is optimized for
the case when each thread has the same index into its scratch space.

Scratch memory is allocated with 64-DWORD granularity and may be [0 - (16M-64)] DWORDs per wave.

3.4. Wave State Registers
The following registers are accessed infrequently, and are only readable/writable via S_GETREG_B32 and
S_SETREG instructions. Some of these registers are read-only, some are writable and others are writable only
when in the trap handler ("PRIV").

Index     Writable? Register
1         Yes          MODE                         Wave mode bits
2         No           STATUS                       Wave status bits
4         PRIV         STATE_PRIV                   wave state bits writable by the trap handler (PRIV=1)
10        No           PERF_SNAPSHOT_DATA           Data captured during performance snapshot
11        No           PERF_SNAPSHOT_PC_LO          PC[31:0] of performance snapshot
12        No           PERF_SNAPSHOT_PC_HI          PC[39:32] of performance snapshot
15        No           PERF_SNAPSHOT_DATA1          Data captured during performance snapshot
16        No           PERF_SNAPSHOT_DATA2          Data captured during performance snapshot
17        PRIV         EXCP_FLAG_PRIV               Exception flags writable only by trap handler
18        Yes          EXCP_FLAG_USER               Exception flags writable by user
19        PRIV         TRAP_CTRL                    Trap exception enables; writable only by trap handler
20        PRIV         SCRATCH_BASE_LO              user-read only; writable only by trap handler
21        PRIV         SCRATCH_BASE_HI              user-read only; writable only by trap handler
23        No           HW_ID1                       read only. debug only - not predictable values
24        No           HW_ID2                       read only. debug only - not predictable values
29        No           SHADER_CYCLES_LO             Get the current shader clock counter value
30        No           SHADER_CYCLES_HI             Get the current shader clock counter value
28        No           IB_STS2

3.4.1. STATUS register
Status register fields can be read but not written to by the shader. These bits are initialized at wave-creation
time or updated during execution.

                                               Table 5. Status Register Fields
Field                       Bit Description
                            Pos
PRIV                        5    Privileged mode. Indicates that the wave is in the trap handler. Gives write access to TTMP
                                 registers.
TRAP_EN                     6    Indicates that a trap handler is present. When set to zero, traps are not taken.
EXPORT_RDY                  8    Pixel shaders only: This status bit indicates that export buffer space has been allocated.
                                 The shader stalls any export instruction until this bit becomes "1". It gets set to 1 when
                                 export buffer space has been allocated. Shader hardware checks this bit before executing any
                                 EXPORT instruction to Position, Z or MRT targets, and put the wave into a waiting state if
                                 the alloc has not yet been received. The alloc arrives eventually (unless SKIP_EXPORT is
                                 set) as a message and the shader then continues with the export.
