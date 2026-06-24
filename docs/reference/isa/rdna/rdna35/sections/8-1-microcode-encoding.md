# 8.1. Microcode Encoding

> RDNA3.5 ISA — pages 87–87

Chapter 8. Scalar Memory Operations
Scalar Memory Loads (SMEM) instructions allow a shader program to load data from memory into SGPRs
through the Constant Cache ("Kcache"). Instructions can load from 1 to 16 DWORDs. Data is loaded directly
into SGPRs without any format conversion.

The scalar unit loads consecutive DWORDs from memory to the SGPRs. This is intended primarily for loading
ALU constants and for indirect T#/S# lookup. No data formatting is supported, nor is byte or short data.

Loads come in two forms: one that simply takes a base-address pointer, and the other that uses a vertex-buffer
constant to provide: base, size and stride.

8.1. Microcode Encoding
Scalar memory load instructions are encoded using the SMEM microcode format.

The fields are described in the table below:

                                       Table 34. SMEM Encoding Field Descriptions
Field     Size Description
OP        8     Opcode. See the next table.
SDATA     7     SGPRs to return Load data to.
                   • Loads of 2 DWORDs must have an even SDST-sgpr.
                   • Loads of 4 or more DWORDs must have their DST-gpr aligned to a multiple of 4.
                   • SDATA must be: SGPR or VCC. Not: EXEC, M0 or NULL except for instructions that return nothing: these
                     may use NULL
SBASE     6     SGPR-pair (SBASE has an implied LSB of zero) that provides a base address, or for BUFFER instructions, a
                set of 4 SGPRs (4-sgpr aligned) that hold the resource constant.
                For BUFFER instructions, the only resource fields used are: base, stride, num_records.
OFFSET 21       Instruction Address Offset : An immediate signed byte offset.
                Negative offsets only work with S_LOAD; a negative offset applied to S_BUFFER results in a MEMVIOL.
SOFFSET 7       SGPR that has the 32-bit unsigned byte offset. May only specify an SGPR, M0 or set to "NULL" to not use
                (offset=0).
GLC       1     Globally Coherent.
DLC       1     Device Coherent.

                        Table 35. SMEM Instructions
Opcode # Name                           Opcode # Name
0             S_LOAD_B32                9          S_BUFFER_LOAD_B64
1             S_LOAD_B64                10         S_BUFFER_LOAD_B128
2             S_LOAD_B128               11         S_BUFFER_LOAD_B256
3             S_LOAD_B256               12         S_BUFFER_LOAD_B512
4             S_LOAD_B512               32         S_GL1_INV
8             S_BUFFER_LOAD_B32         33         S_DCACHE_INV
