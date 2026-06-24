# 8.1. Microcode Encoding

> RDNA4 ISA — pages 107–107

Chapter 8. Scalar Memory Operations
Scalar Memory Loads (SMEM) instructions allow a shader program to load data from memory into SGPRs
through the Constant Cache ("Kcache"). Instructions can load from 1 to 16 DWORDs. Data is loaded directly
into SGPRs without any format conversion (no data formatting is supported).

The scalar unit loads consecutive DWORDs from memory to the SGPRs. One common use for SMEM loads is
for loading ALU constants and for indirect T#/S#/V# lookup.

Loads come in two forms: one that simply takes a base-address pointer, and the other that uses a buffer
resource (V#) to provide: base, size and stride.

8.1. Microcode Encoding
Scalar memory load instructions are encoded using the SMEM microcode format.

The fields are described in the table below:

                                       Table 44. SMEM Encoding Field Descriptions
Field             Size    Description
OP                6       Opcode. See the next table.
SDATA             7       SDATA specifies the SGPRs to load the data into.
                            • Loads of 2 DWORDs must have an even SDST-sgpr.
                            • Loads of 3 or more DWORDs must have their SDST-gpr aligned to a multiple of 4 SGPRs.
                            • SDATA must be: SGPR or VCC. Not: EXEC, or M0. NULL is allowed
SBASE             6       SGPR-pair (SBASE has an implied LSB of zero) that provides a base address, or for BUFFER
                          instructions, a set of 4 SGPRs (4-sgpr aligned) that hold the buffer resource.
                          For BUFFER instructions, the only resource fields used are: base, stride, num_records.
IOFFSET           24      Instruction Address Offset : An immediate signed byte offset.
                          Negative IOFFSETs only work with S_LOAD; a negative IOFFSETs applied to S_BUFFER results
                          in a MEMVIOL.
SOFFSET           7       SGPR that has the 32-bit unsigned byte offset. May only specify an SGPR, M0 or set to "NULL" to
                          not use (offset=0).
SCOPE             2       Memory Scope
TH                2       Memory Temporal Hint

See Cache Controls: SCOPE and Temporal-Hint for more information about SCOPE and TH bits.

                          Table 45. SMEM Instructions
Opcode # Name                           Opcode # Name
0          S_LOAD_B32                   19          S_BUFFER_LOAD_B256
1          S_LOAD_B64                   20          S_BUFFER_LOAD_B512
2          S_LOAD_B128                  21          S_BUFFER_LOAD_B96
3          S_LOAD_B256                  24          S_BUFFER_LOAD_I8
4          S_LOAD_B512                  25          S_BUFFER_LOAD_U8
