# 3.4.1. Status register

> RDNA3.5 ISA — pages 29–30

     B128 and B96: 16 byte aligned

If the alignment mode is set to "unaligned", the LDS disables its auto-alignment and doesn’t report error for
misaligned reads & writes.

            if (sh_alignment_mode == unaligned)          align = 0xffff
            else if (B32)                                align = 0xfffC
            else if (B64)                                align = 0xfff8
            else if (B96 or B128)                        align = 0xfff0
            LDSaddr = (addr + offset) & align

3.4. Wave State Registers
The following registers are accessed infrequently, and are only readable/writable via S_GETREG and S_SETREG
instructions. Some of these registers are read-only, some are writable and others are writable only when in the
trap handler ("PRIV").

Code       Register
0          Reserved
1          MODE                       read / write
2          STATUS                     read / write. Only writable when priv=1
3          TRAPSTS                    read / write
14         FLUSH_IB                   write-only. Writing this causes all waves to flush their instruction buffers
15         SH_MEM_BASES               read-only. Allows a wave to read the value of this register to do aperture checks and
                                      memory space conversions. Bits [15:0] = Private Base; [31:16] = Shared Base.
20         FLAT_SCRATCH_LO            read only (writable only while in trap handler)
21         FLAT_SCRATCH_HI            read only (writable only while in trap handler)
23         HW_ID1                     read only. debug only - not predictable values
24         HW_ID2                     read only. debug only - not predictable values
29         SHADER_CYCLES              Get the current graphics clock counter value

3.4.1. Status register
Status register fields can be read but not written to by the shader. While in the trap handler, certain STATUS fields
can be written. These bits are initialized at wave-creation time. The table below describes the status register
fields.

                                              Table 5. Status Register Fields
Field                       Bit Write Description
                            Pos when
                                Priv?
SCC                         0   Y     Scalar condition code. Used as a carry-out bit. For a comparison instruction, this bit
                                      indicates failure or success. For logical operations, this is 1 if the result is non-zero.
SYS_PRIO                    2:1 Y     Wave priority set at wave creation time. See S_SETPRIO instruction for details. 0 is
                                      lowest, 3 is highest priority.
USER_PRIO                   4:3 Y     Wave’s priority set by shader program itself. See S_SETPRIO instruction for details.

Field                       Bit Write Description
                            Pos when
                                Priv?
PRIV                        5    N    Privileged mode. Indicates that the wave is in the trap handler. Gives write access to
                                      TTMP registers.
TRAP_EN                     6    N    Indicates that a trap handler is present. When set to zero, traps are not taken.
EXPORT_RDY                  8    Y    This status bit indicates if export buffer space has been allocated. The shader stalls
                                      any export instruction until this bit becomes "1". It gets set to 1 when export buffer space
                                      has been allocated.
                                      Shader hardware checks this bit before executing any EXPORT instruction to
                                      Position, Z or MRT targets, and put the wave into a waiting state if the alloc has not
                                      yet been received. The alloc arrives eventually (unless SKIP_EXPORT is set) as a
                                      message and the shader then continues with the export.
EXECZ                       9    N    Exec Mask is Zero.
VCCZ                        10   N    Vector Condition Code is Zero.
IN_WG                       11   N    Wave is a member of a work-group of more than one wave.
IN_BARRIER                  12   N    Wave is waiting at a barrier.
HALT                        13   Y    Wave is halted or scheduled to halt.
                                      HALT can be set by the host via wave-control messages, or by the shader. The HALT
                                      bit is ignored while in the trap handler (PRIV = 1). HALT is also ignored if a host-
                                      initiated trap is received (request to enter the trap handler).
TRAP                        14   N    Wave is flagged to enter the trap handler as soon as possible.
VALID                       16   N    Wave is valid (has been created and not yet ended)
SKIP_EXPORT                 18   Y    For Pixel and Vertex Shaders only.
                                      "1" means this shader is not allocated export buffer space, so export instructions are
                                      ignored (treated as NOPs). For pixel shaders, this is set to 1 when both the
                                      COL0_EXPORT_FORMAT and Z_EXPORT_FORMAT are set to ZERO. If
                                      SKIP_EXPORT==1, Must_export must be zero and vice versa.
PERF_EN                     19   N    Performance counters are enabled for this wave
CDBG_USER                   20   Y    User-controlled conditional debug. Set at wave-create time by a user register. Can be
                                      used in conditional branches.
CDBG_SYS                    21   Y    System-controlled conditional debug. Set at wave-create time by a system register.
                                      Can be used in conditional branches.
FATAL_HALT                  23   N    Indicates that the wave has halted due to a fatal error:
                                      illegal instruction . The difference between halt and fatal_halt is that fatal_halt stops
                                      waves even when PRIV=1.
NO_VGPRS                    24   N    Indicates that this wave has released all of its VGPRs.
LDS_PARAM_RDY               25   Y    PS shaders only: indicates that LDS has been written with vertex attribute data and
                                      the shader may now execute LDS_PARAM_LOAD instructions. If the wave attempts to
                                      issue LDS_PARAM_LOAD before this bit is set, it stalls until the bit is set.
MUST_GS_ALLOC               26   N    GS shader must issue a GS_ALLOC_REQ message before terminating.
                                      Sending this message clears this bit.
MUST_EXPORT                 27   Y    PS: this wave must export color ("export-done") before it terminates.
                                      Set to 1 for PS waves unless "skip_export==1". Cleared when PS exports data with
                                      export’s Done bit set to 1.
                                      GS: this wave must perform a GDS_ordered_count before terminating. Cleared when
                                      a GS shader issues a GDS_ordered_count. GS is initialized to 1 normally, but to zero
                                      for "no export" passes (stream-out only).
IDLE                        28   N    Wave is idle (has no outstanding instructions). Used by the host (GRBM) to
                                      determine if a wave is valid, halted and idle - able to read other wave state.
SCRATCH_EN                  29   Y    Indicate that the wave has scratch memory allocated. This bit gets set to 1 if the wave
                                      has FLAT_SCRATCH initialized; otherwise is zero.
