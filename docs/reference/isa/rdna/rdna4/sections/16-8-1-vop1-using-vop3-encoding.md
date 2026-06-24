# 16.8.1. VOP1 using VOP3 encoding

> RDNA4 ISA — pages 346–346

Input modifiers are ignored. When used in the VOP1 encoding the usual OPSEL16 rules apply.

16.8.1. VOP1 using VOP3 encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x180.
