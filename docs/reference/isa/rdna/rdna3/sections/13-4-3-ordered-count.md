# 13.4.3. Ordered Count

> RDNA3 ISA — pages 143–144

 5. INF + (float, +0, -0) = INF, with infinity sign preserved
 6. NaN + NaN = SRC0’s NaN, converted to QNaN

13.4. Global Wave Sync & Atomic Ordered Count
Global Wave Sync (GWS) provides a capability to synchronize between different waves across the entire GPU.
GWS instructions use LGKMcnt to determine when the operation has completed.

13.4.1. GWS and Ordered Count Programming Rule
"GWS" instructions (ordered count and GWS*) must be issued as a single instruction clause of the form:

   S_WAITCNT LGKMcnt==0 // this is only necessary if there might be any outstanding GDS instructions
   GWS_instruction
   S_WAITCNT LGKMcnt==0
   <any instruction except: S_ENDPGM (pad with NOP if the next instruction is s_endpgm)

Before issuing a GWS or Ordered Count instruction, the user must make sure that there are no outstanding GDS
instructions. Failure to do this may cause a "NACK" to arrive out of order.

  Programming Rule:         the source and destination VGPRs in a GWS or ordered count instruction must not
                            be the same. When an ordered count operation is NACK’d, the destination VGPR
                            may be written with data. If this VGPR is the same as the source VGPR, that
                            prevents the instruction from being replayed later if it was interrupted due to a
                            context switch.

13.4.2. EXEC Handling
GDS / GWS is now only a single lane wide. If the EXEC mask has more than one bit set to 1, hardware behaves
as if only EXEC had only one "1" in it: the least significant one. GDS / GWS opcodes are not skipped when
EXEC==0.

For these opcodes, if EXEC==0, the hardware acts as if EXEC==0…001 for the instruction:
    ORDERED_COUNT / GWS_INIT / SEMA_BR/GWS_BARRIER

For other GDS / GWS opcodes, the instruction is sent with EXE==0, nothing is sent to or returned from
GDS/GWS. In hardware, data is sent but it is ignored and data is returned and ignored in order to keep LGKMcnt
working.

13.4.3. Ordered Count
Ordered count generates a pointer in wave-creation order to an append buffer of unlimited size.

Ordered Alloc generates a pointer to a ring buffer of finite size which is returned to the wave in "VDST". The
ordered alloc counter can be issued up to 4 times from a shader. Ordered count and alloc use the same
instruction - the difference is in how the GDS counters are initialized with their config registers.

The GDS unit supports an instruction that operates on dedicated append/consume counters:
  • DS_ORDERED_COUNT Takes one value from the first valid lane and sends to GDS.

For shaders that use this function, this instruction must be issued once and only once per wave. The GDS
receives these in arbitrary order from different waves across the chip, but processes them in the order the
waves were created. The GDS contains a large fifo to hold these pending requests.

Instruction Fields

Field           Normal GDS              GDS Ordered Count                                Global Wave Sync (GWS)
OP              any GDS op              DS_ORDERED_COUNT*                                GWS_INIT, GWS_SEMA_V,
                                                                                         GWS_SEMA_BR, GWS_SEMA_P
                                                                                         GWS_SEMA_RELEASE_ALL,
                                                                                         GWS_BARRIER
GDS             1                       1                                                1
VDST            VGPR to write result    VGPR to write result to                          unused
                to
ADDR            VGPR which supplies Increment, from the first valid data.                Used for: barrier, init and
                byte address offset If no valid data, increment=0.                       sema_br;
                                                                                         unused for others.
DATA0           VGPR which supplies unused                                               unused
                first data source
DATA1           VGPR which supplies unused                                               unused
                second data source
Offset0[7:0]    Same usage as LDS       Ordered Count Index.                             { 0,0,resource_index[5:0] }
                                        Must be multiple of 4 (2 LSB’s must be zero)
Offset1[0]      Same usage as LDS       wave_release                                     unused
Offset1[1]      Same usage as LDS       wave_done                                        unused
Offset1[3:2]    Same usage as LDS       unused                                           unused
Offset1[5:4]    Same usage as LDS       ordered-index-opcode :                           unused
                                        0 = Add (ds_add_rtn_b32)
                                        1 = Exchange (ds_wrxchg_rtn_b32)
                                        2 = Reserved
                                        3 = Wrap (ds_wrap_rtn_b32)
Offset1[7:6]    Same usage as LDS       unused                                           unused
M0[15:0]        gds_size[15:0] in bytes { waveCrawlerInc[2:0], logicalWaveID[12:0] }     unused
                                        In graphics pipe, logicalWaveID[2:0] is really
                                        packerID
M0[31:16]       gds_base[15:0] in       orderedCntBase[15:0]                             { 10'0, gds_base[5:0] }
                bytes                   Ordered count base is in DWORDs.                 gdsBase = resourceBase
                                        (2 LSB’s are ignored, forced to zero - DWORD
                                        aligned)

ORDERED COUNT Targets
     The OFFSET0[5:2] field of ordered-count instructions reference one of 16 registers in GDS. These are listed
     in the GDS section: GS NGG Streamout Instructions. See: GS NGG Streamout Instructions Only the ADD
     instruction may be used on targets that are 64 bits (offset[5:2] = 8 - 15).
     Exchange can only be used with offset[5:2] = 4 - 7.

APPEND and CONSUME
     Append and Consume count bits in EXEC and add or subtract the count from the GDS stored value. GDS
     now only operates on a single lane, but for Append & Consume the full EXEC mask is still considered.
