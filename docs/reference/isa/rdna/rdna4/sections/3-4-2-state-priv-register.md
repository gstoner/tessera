# 3.4.2. STATE_PRIV register

> RDNA4 ISA — pages 33–33

Field                       Bit Description
                            Pos
EXECZ                       9     Exec Mask is Zero.
VCCZ                        10    Vector Condition Code is Zero.
IN_WG                       11    Wave is a member of a work-group of more than one wave.
TRAP                        14    Wave is flagged to enter the trap handler as soon as possible.
TRAP_BARRIER_               15    Indicates that the trap-barrier has completed but wave has not yet waited on that barrier
COMPLETE
VALID                       16    Wave is valid (has been created and not yet ended)
SKIP_EXPORT                 18    Pixel and Vertex Shaders only:
                                  "1" means this shader is not allocated export buffer space, so export instructions are
                                  ignored (treated as NOPs). For pixel shaders, this is set to 1 when both the
                                  COL0_EXPORT_FORMAT and Z_EXPORT_FORMAT are set to ZERO. If SKIP_EXPORT==1,
                                  Must_export must be zero and vice versa.
FATAL_HALT                  23    Indicates that the wave has halted due to a fatal error:

                                    • illegal instruction

                                  The difference between halt and fatal_halt is that fatal_halt stops waves even when
                                  PRIV=1.
NO_VGPRS                    24    Indicates that this wave has released all of its VGPRs.
LDS_PARAM_RDY               25    Pixel shaders only: indicates that LDS has been written with vertex attribute data and the
                                  shader may now execute DS_PARAM_LOAD instructions. If the wave attempts to issue
                                  DS_PARAM_LOAD before this bit is set, it stalls until the bit is set.
MUST_GS_ALLOC               26    Indicates that the GS shader must issue a GS_ALLOC_REQ message before terminating.
                                  Sending this message clears this bit.
MUST_EXPORT                 27    Pixel Shaders: this wave must export color ("export-done") before it terminates.
                                  Set to 1 for PS waves unless "skip_export==1". Cleared when PS exports data with export’s
                                  Done bit set to 1.
                                  Is set to zero for other wave types.
IDLE                        28    Wave is idle (has no outstanding instructions). Used via register-read to determine if a
                                  wave is valid, halted and idle - able to read other wave state.
WAVE64                      29    Wave is 64 (0 = wave32)
DYN_VGPR_EN                 30    Indicates that the wave is running using Dynamic VGPRs.

3.4.2. STATE_PRIV register
STATE_PRIV register fields can be read by the user shader and written while in the trap handler (PRIV=1).
These bits are initialized at wave-creation time or updated during execution.

                                            Table 6. STATE_PRIV Register Fields
Field                       Bit     Description
                            Pos
WG_RR_EN                    0       Workgroup Round-robin arbitration requested.
SLEEP_WAKEUP                1       The wave has one credit to skip the next S_SLEEP instruction due to previously receiving
                                    a S_WAKEUP
BARRIER_COMPLETE            2       has the barrier completed but wave has not yet waited on that barrier
SCC                         9       Scalar condition code. Used as a carry-out bit. For a comparison instruction, this bit
                                    indicates failure or success. For logical operations, this is 1 if the result is non-zero.
