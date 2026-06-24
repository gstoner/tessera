# 9.3. Buffer Data

> RDNA3 ISA — pages 95–95

Instruction                               VGPR Format                               Memory        Notes
                                                                                    Format
BUFFER_STORE_FORMAT_X                     float, uint or sint                       FORMAT field data type in VGPR is
                                          data in V0[31:0]                                       based on FORMAT
BUFFER_STORE_D16_FORMAT_X                 float, ushort or sshort                                field.
                                          data in V0[15:0]
BUFFER_STORE_D16_FORMAT_XY                float, ushort or sshort
                                          data in V0[15:0], V0[31:16]
BUFFER_STORE_D16_FORMAT_XYZ               float, ushort or sshort
                                          data in V0[15:0], V0[31:16], V1[15:0]
BUFFER_STORE_D16_FORMAT_XYZW              float, ushort or sshort
                                          data in V0[15:0], V0[31:16], V1[15:0],
                                          V1[31:16]
BUFFER_STORE_D16_HI_FORMAT_X              float, ushort or sshort
                                          data in V0[31:16]

9.3. Buffer Data
The amount and type of data that is loaded or stored is controlled by the following: the resource format field,
destination-component-selects (dst_sel), and the opcode.

Data-format can come from the resource, instruction fields, or the opcode itself. MTBUF derives data-format
from the instruction, MUBUF-"format" instructions use format from the resource, and other MUBUF opcode
derive data-format from the instruction itself.

DST_SEL comes from the resource, but is ignored for many operations.

                                             Table 42. Buffer Instructions
                               Instruction                          Data Format    DST SEL
                               TBUFFER_LOAD_FORMAT_*                instruction    identity
                               TBUFFER_STORE_FORMAT_*               instruction    identity
                               BUFFER_LOAD_<type>                   derived        identity
                               BUFFER_STORE_<type>                  derived        identity
                               BUFFER_LOAD_FORMAT_*                 resource       resource
                               BUFFER_STORE_FORMAT_*                resource       resource
                               BUFFER_ATOMIC_*                      derived        identity

Instruction : The instruction’s format field is used instead of the resource’s fields.

Data format derived : The data format is derived from the opcode and ignores the resource definition. For
example, BUFFER_LOAD_U8 sets the data-format to uint-8.

                   The resource’s data format must not be INVALID; that format has specific meaning
                  (unbound resource), and for that case the data format is not replaced by the instruction’s
                   implied data format.

DST_SEL identity : Depending on the number of components in the data-format, this is: X000, XY00, XYZ0, or
XYZW.
