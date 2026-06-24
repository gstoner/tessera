# 16.15.1. LDS Instruction Limitations

> RDNA3.5 ISA — pages 581–581

  RETURN_DATA[31 : 0] = MEM[ADDR + OFFSET.u32].b32;
  RETURN_DATA[63 : 32] = MEM[ADDR + OFFSET.u32 + 4U].b32;
  RETURN_DATA[95 : 64] = MEM[ADDR + OFFSET.u32 + 8U].b32;
  RETURN_DATA[127 : 96] = MEM[ADDR + OFFSET.u32 + 12U].b32

16.15.1. LDS Instruction Limitations
Some of the DS instructions are available only to GDS, not LDS. These are:

  • DS_GWS_SEMA_RELEASE_ALL
  • DS_GWS_INIT
  • DS_GWS_SEMA_V
  • DS_GWS_SEMA_BR
  • DS_GWS_SEMA_P
  • DS_GWS_BARRIER
  • DS_ORDERED_COUNT
