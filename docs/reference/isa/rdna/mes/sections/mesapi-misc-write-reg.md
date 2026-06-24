# MESAPI_MISC__WRITE_REG

> Micro Engine Scheduler ISA — pages 43–43

             };

             enum MESAPI_MISC_OPCODE
             {
                   MESAPI_MISC__WRITE_REG,
                   MESAPI_MISC__INV_GART,
                   MESAPI_MISC__QUERY_STATUS,
                   MESAPI_MISC__READ_REG,
                   MESAPI_MISC__WAIT_REG_MEM,
                   MESAPI_MISC__SET_SHADER_DEBUGGER,
                   MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE,
                   MESAPI_MISC__NOTIFY_TO_UNMAP_PROCESSES,

                   MESAPI_MISC__MAX,
             };

                    •   opcode - Changes functionality based on what MESAPI_MISC_OPCODE is used. See
                        each opcode's section for more details

MESAPI_MISC__WRITE_REG
                 Perform register write on request of KMD.
             struct WRITE_REG
             {
                   uint32_t                      reg_offset;
                   uint32_t                      reg_value;
             };

                    •   reg_offset - Offset of the register

                    •   reg_value - Value to be written to the register

MESAPI_MISC__INV_GART
                 Perform GART invalidation.
             struct INV_GART
             {
                   uint64_t                      inv_range_va_start;
                   uint64_t                      inv_range_size;
             };

                    •   inv_range_va_start - starting VA for invalidation range

April 2024                                                                                         43
